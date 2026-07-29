"""
소통혁신24 수상작 설명 수집 스크립트 (v2 - AJAX 엔드포인트 방식)
================================================================

구조 (DevTools로 확인한 실제 동작):
  1) GET  /front/epilogue/epilogueNewViewPage.do?...   -> 껍데기 페이지 (JSESSIONID, CSRF)
  2) POST /front/epilogue/epilogueNewView.do           -> 작품 카드 목록 HTML 조각
       form: bbs_id, new_bbs_id, pagetype, progress, prevew_value
       headers: Ajax:true, X-Requested-With:XMLHttpRequest, X-Csrf-Token
  3) 조각 안의 <input id="wrkCn{wrk_id}" value="..."> -> 작품 설명

  페이지의 viewCnddtWrkCn(wrk_id) 가 하는 일:
      $('#popupWrkCn_content').html($('#wrkCn'+wrk_id).val());

사용법:
  # 1단계: 세종시 1페이지 테스트 (DB 변경 없음)
  python collect_work_descriptions.py --dry-run --debug --url "https://sotong.go.kr/front/epilogue/epilogueNewViewPage.do?menu_id=528&bbs_id=664c0384f8464eecb6dab9dd70c9b8c3&pagetype=cnddt"

  # 2단계: 전체 실행
  python collect_work_descriptions.py
"""

import argparse
import os
import random
import re
import sys
import time
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne

# ---------------------------------------------------------------- 설정

try:
    from dotenv import load_dotenv

    for candidate in ("backend/.env", ".env", "../backend/.env"):
        if os.path.exists(candidate):
            load_dotenv(candidate)
            print(f"[env] {candidate} 로드")
            break
except ImportError:
    pass

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB = os.getenv("MONGODB_DB", "ai_review_board")
if not MONGODB_URL:
    sys.exit("MONGODB_URL 이 없습니다. backend/.env 확인")

COLLECTION = "contest_works"

BASE = "https://sotong.go.kr"
AJAX_URL = f"{BASE}/front/epilogue/epilogueNewView.do"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

WRK_INPUT_ID = re.compile(r"^wrkCn([0-9a-fA-F]{32})$")

CSRF_PATTERNS = [
    re.compile(r'name=["\']_csrf["\'][^>]*(?:value|content)=["\']([^"\']+)', re.I),
    re.compile(r'(?:value|content)=["\']([0-9a-f\-]{30,40})["\'][^>]*name=["\']_csrf', re.I),
    re.compile(r'csrf[_\-]?token["\']?\s*[:=]\s*["\']([0-9a-f\-]{30,40})', re.I),
    re.compile(r'["\']([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})["\']'),
]

TIMEOUT = 20
RETRY = 3


# ---------------------------------------------------------------- 유틸


def html_to_text(raw):
    """
    모달 내용 HTML -> 평문.
    <span>이 잘게 쪼개져 있어 구분자를 넣으면 문장이 깨지므로
    <br>/<p>/<div>/<li> 만 줄바꿈으로 바꾸고 나머지는 그대로 이어붙인다.
    """
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "li", "tr"]):
        blk.append("\n")

    text = soup.get_text("")          # 구분자 없이 이어붙임
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_csrf(html_doc):
    for pat in CSRF_PATTERNS:
        m = pat.search(html_doc)
        if m:
            return m.group(1)
    return None


def parse_url_params(url):
    q = parse_qs(urlparse(url).query)
    return {
        "bbs_id": (q.get("bbs_id") or [""])[0],
        "pagetype": (q.get("pagetype") or ["cnddt"])[0],
    }


def extract_works(fragment):
    """
    작품 설명은 <textarea id="wrkCn{wrk_id}"> 안에 HTML이 escape 되어 들어있음.
    (jQuery .val() 이 textarea 내용을 읽는 구조)
    혹시 모를 구조 변화 대비로 input[value] 도 함께 처리.
    """
    soup = BeautifulSoup(fragment, "html.parser")
    out = {}

    for tag in soup.find_all(["textarea", "input"]):
        m = WRK_INPUT_ID.match(tag.get("id") or "")
        if not m:
            continue
        wrk_id = m.group(1)

        if tag.name == "textarea":
            # get_text()가 &lt;p&gt; 같은 엔티티를 실제 태그 문자열로 복원
            raw_html = tag.get_text()
        else:
            raw_html = tag.get("value") or ""

        if not raw_html.strip():
            continue

        title_el = soup.find(id="wrkCn_title_" + wrk_id)
        out[wrk_id] = {
            "title": title_el.get_text(" ", strip=True) if title_el else "",
            "html": raw_html,
            "text": html_to_text(raw_html),
        }
    return out


# ---------------------------------------------------------------- 수집


def fetch_page_works(session, view_url, debug=False):
    params = parse_url_params(view_url)
    if not params["bbs_id"]:
        print("    [건너뜀] bbs_id 없음")
        return {}

    # 1) 껍데기 페이지 GET -> 세션 쿠키 + CSRF
    try:
        r0 = session.get(view_url, timeout=TIMEOUT)
        r0.raise_for_status()
        r0.encoding = "utf-8"
    except Exception as exc:
        print("    [GET 실패] " + str(exc))
        return {}

    csrf = find_csrf(r0.text)

    headers = {
        "Ajax": "true",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": view_url,
        "Origin": BASE,
        "Accept": "text/html, */*; q=0.01",
    }
    if csrf:
        headers["X-Csrf-Token"] = csrf

    # progress 값은 공고 단계마다 다를 수 있어 후보를 순회
    for progress in ["5", "4", "3", "2", "1", ""]:
        data = {
            "bbs_id": params["bbs_id"],
            "new_bbs_id": "",
            "pagetype": params["pagetype"],
            "progress": progress,
            "prevew_value": "",
        }

        resp = None
        for attempt in range(1, RETRY + 1):
            try:
                resp = session.post(AJAX_URL, data=data, headers=headers, timeout=TIMEOUT)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                break
            except Exception as exc:
                resp = None
                if attempt == RETRY:
                    print("    [POST 실패] progress=" + progress + " " + str(exc))
                else:
                    time.sleep(attempt * 2)

        if resp is None:
            continue

        works = extract_works(resp.text)

        if debug:
            fn = "debug_" + params["bbs_id"][:8] + "_p" + (progress or "none") + ".html"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print("    [debug] {} 저장 ({}자, wrkCn {}건, csrf={})".format(
                fn, len(resp.text), len(works), "O" if csrf else "X"))

        if works:
            return works

    return {}


# ---------------------------------------------------------------- 메인


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", type=str, default="")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=8000)
    col = client[MONGODB_DB][COLLECTION]
    print("[DB] {}.{} — 전체 {}건\n".format(MONGODB_DB, COLLECTION, col.count_documents({})))

    if args.url:
        urls = [args.url]
    else:
        urls = sorted(u for u in col.distinct("source_url") if u)
    if args.limit:
        urls = urls[: args.limit]
    print("[대상] 페이지 {}개\n".format(len(urls)))

    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"})

    stat = {"ok": 0, "zero": 0, "parsed": 0, "matched": 0,
            "updated": 0, "empty": 0, "unmatched": []}

    for idx, url in enumerate(urls, 1):
        print("[{}/{}] {}".format(idx, len(urls), url[:92]))

        works = fetch_page_works(session, url, debug=args.debug)

        if not works:
            stat["zero"] += 1
            print("    추출 0건")
            time.sleep(args.sleep)
            continue

        stat["ok"] += 1
        stat["parsed"] += len(works)
        print("    추출 {}건".format(len(works)), end="")

        ops = []
        for wrk_id, d in works.items():
            text = d["text"]
            if not text:
                stat["empty"] += 1
                continue
            if col.count_documents({"wrk_id": wrk_id}, limit=1) == 0:
                stat["unmatched"].append(wrk_id)
                continue
            stat["matched"] += 1

            if args.dry_run:
                print("\n      · {} ({}자) {}".format(wrk_id[:8], len(text), text[:100]))
                continue

            ops.append(UpdateOne(
                {"wrk_id": wrk_id},
                {"$set": {
                    "description": text,
                    "description_html": d["html"],
                    "description_source": "sotong_modal",
                }},
            ))

        if ops:
            res = col.bulk_write(ops, ordered=False)
            stat["updated"] += res.modified_count
            print(" → 갱신 {}건".format(res.modified_count))
        else:
            print()

        time.sleep(args.sleep + random.uniform(0, 0.4))

    print("\n" + "=" * 56)
    print("  결과 요약")
    print("=" * 56)
    print("  성공 페이지     : {}".format(stat["ok"]))
    print("  0건 페이지      : {}".format(stat["zero"]))
    print("  추출한 작품     : {}건".format(stat["parsed"]))
    print("  DB 매칭 성공    : {}건".format(stat["matched"]))
    print("  설명 비어있음   : {}건".format(stat["empty"]))
    print("  DB에 없는 id    : {}건".format(len(stat["unmatched"])))
    if args.dry_run:
        print("\n  ** dry-run — DB 변경 없음 **")
    else:
        print("  실제 갱신       : {}건".format(stat["updated"]))
    if stat["unmatched"]:
        print("\n  미매칭 샘플: {}".format(stat["unmatched"][:5]))

    client.close()


if __name__ == "__main__":
    main()

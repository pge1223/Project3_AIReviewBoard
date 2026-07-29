"""
oss.kr 오픈소스 개발자대회 수상작 크롤러
- sourceCategory=855 (개발자대회) 필터
- 목록 페이지 파싱 → 상세 페이지 크롤링
- MongoDB oss_winners 컬렉션 저장
"""

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne
import time
import random
import re
from datetime import datetime

MONGODB_URL = "mongodb://reviewboard_admin:reviewboard2026!@localhost:27018/ai_review_board?authSource=admin"
DB_NAME = "ai_review_board"
COLLECTION_NAME = "oss_winners"
BASE_URL = "https://oss.kr"
LIST_URL = f"{BASE_URL}/pages/8"
DETAIL_URL = f"{BASE_URL}/opensource/hub"

client = MongoClient(MONGODB_URL)
col = client[DB_NAME][COLLECTION_NAME]
col.create_index("project_id", unique=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": BASE_URL,
})


def fetch_list_page(page: int):
    """목록 페이지에서 프로젝트 ID + 기본 정보 파싱"""
    r = session.get(LIST_URL, params={
        "pageIndex": page,
        "sourceCategory": 855,
        "sourceTags": 855,
    }, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    items = []
    for card in soup.select("div.component-card[data-project-id]"):
        project_id = card.get("data-project-id", "")
        title = card.select_one("div.card-title")
        description = card.select_one("p.card-description")
        tags = [t.get_text(strip=True).lstrip("#") for t in card.select("span.tag")]
        pts_text = card.get_text()
        pts_match = re.search(r"(\d+)\s*pts", pts_text)

        items.append({
            "project_id": project_id,
            "title": title.get_text(strip=True) if title else "",
            "description": description.get_text(strip=True) if description else "",
            "tags": tags,
            "pts": int(pts_match.group(1)) if pts_match else 0,
        })

    # 총 페이지 수 파악
    total_el = soup.select_one("span.total-count, .result-count, [class*='total']")
    total_text = soup.get_text()
    total_match = re.search(r"총\s*([\d,]+)\s*건", total_text)
    total_count = int(total_match.group(1).replace(",", "")) if total_match else 0

    return items, total_count


def fetch_detail(project_id: str) -> dict:
    """상세 페이지에서 GitHub URL, 기술분류, 연도, 개발자 정보 파싱"""
    url = f"{DETAIL_URL}/{project_id}"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [WARN] 상세 페이지 오류 {project_id}: {e}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    detail = {}

    # 테이블 형태로 key-value 파싱
    rows = soup.select("table tr, dl dt, .info-row")
    page_text = soup.get_text(separator="\n")

    # GitHub 저장소 URL
    github_link = soup.find("a", href=re.compile(r"github\.com"))
    if github_link:
        detail["github_url"] = github_link.get("href", "")

    # 각 필드 파싱 (라벨-값 패턴)
    for label_el in soup.find_all(string=re.compile(r"기술분류|연도|산업분류|사업분류|개요|국내외")):
        parent = label_el.find_parent()
        if parent:
            next_el = parent.find_next_sibling()
            if next_el:
                text = next_el.get_text(strip=True)
                if "기술분류" in label_el:
                    detail["tech_category"] = text
                elif "연도" in label_el:
                    detail["year"] = text
                elif "산업분류" in label_el:
                    detail["industry"] = text
                elif "사업분류" in label_el:
                    detail["biz_category"] = text
                elif "개요" in label_el:
                    detail["summary"] = text
                elif "국내외" in label_el:
                    detail["domestic_yn"] = text

    # 수상 등급 파싱 (개요에서 추출)
    desc = detail.get("summary", "")
    award_match = re.search(r"(대상|최우수상|우수상|동상|장려상|은상|금상|특별상|입선)\s*[\(\（]?(학생|일반|기업)?[\)\）]?", desc)
    if award_match:
        detail["award_grade"] = award_match.group(0).strip()

    # 수상 연도 파싱 (개요에서 추출)
    year_match = re.search(r"(\d{4})년", desc)
    if year_match:
        detail["award_year"] = year_match.group(1)

    # 주요 개발자 GitHub
    dev_links = []
    for a in soup.select("table a[href*='github.com']"):
        href = a.get("href", "")
        if href and href != detail.get("github_url"):
            dev_links.append(href)
    if dev_links:
        detail["developer_github"] = dev_links

    detail["detail_url"] = f"{DETAIL_URL}/{project_id}"
    return detail


def crawl():
    print("=== oss.kr 오픈소스 개발자대회 수상작 크롤러 시작 ===")

    # 1페이지로 총 건수 파악
    first_items, total_count = fetch_list_page(1)
    per_page = len(first_items)
    total_pages = (total_count + per_page - 1) // per_page if per_page > 0 else 1
    print(f"총 {total_count}건 / 페이지당 {per_page}건 / 총 {total_pages}페이지")

    all_items = list(first_items)

    for page in range(2, total_pages + 1):
        time.sleep(random.uniform(0.5, 1.0))
        items, _ = fetch_list_page(page)
        if not items:
            print(f"페이지 {page}: 결과 없음 — 종료")
            break
        all_items.extend(items)
        print(f"페이지 {page}/{total_pages}: {len(items)}건 (누적 {len(all_items)}건)")

    print(f"\n목록 수집 완료: {len(all_items)}건")

    # 상세 페이지 크롤링 + 저장
    print("\n상세 페이지 크롤링 + DB 저장...")
    ops = []
    for i, item in enumerate(all_items, 1):
        time.sleep(random.uniform(0.3, 0.7))
        detail = fetch_detail(item["project_id"])

        doc = {
            "project_id": item["project_id"],
            "title": item["title"],
            "description": item["description"],
            "tags": item["tags"],
            "pts": item["pts"],
            "github_url": detail.get("github_url", ""),
            "developer_github": detail.get("developer_github", []),
            "tech_category": detail.get("tech_category", ""),
            "industry": detail.get("industry", ""),
            "biz_category": detail.get("biz_category", ""),
            "summary": detail.get("summary", ""),
            "award_grade": detail.get("award_grade", ""),
            "award_year": detail.get("award_year", ""),
            "year": detail.get("year", ""),
            "detail_url": detail.get("detail_url", ""),
            "source_site": "oss.kr",
            "crawled_at": datetime.utcnow().isoformat(),
        }

        ops.append(UpdateOne(
            {"project_id": item["project_id"]},
            {"$set": doc},
            upsert=True,
        ))

        print(f"  [{i}/{len(all_items)}] {item['title']} | {detail.get('award_grade', '?')} | GitHub: {detail.get('github_url', 'X')}")

        if len(ops) >= 10 or i == len(all_items):
            result = col.bulk_write(ops)
            print(f"  → 저장: upserted={result.upserted_count}, modified={result.modified_count}")
            ops = []

    total = col.count_documents({})
    print(f"\n=== 완료! 총 {total}건 저장 ===")

    # 샘플 출력
    print("\n[샘플 3건]")
    for doc in col.find({}, {"_id": 0, "project_id": 1, "title": 1, "award_grade": 1, "github_url": 1}).limit(3):
        print(f"  - [{doc['project_id']}] {doc['title']} | {doc.get('award_grade', '?')} | {doc.get('github_url', 'X')}")


if __name__ == "__main__":
    crawl()

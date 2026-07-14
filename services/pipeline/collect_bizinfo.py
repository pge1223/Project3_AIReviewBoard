import requests
import json
import time
from datetime import datetime
from pymongo import MongoClient
from pathlib import Path
import xml.etree.ElementTree as ET
import re

BIZINFO_API_KEY = "P3i5Z9"
BASE_URL = "https://www.bizinfo.go.kr/uss/rss/bizinfoApi.do"
SAVE_DIR = Path("/mnt/hdd/sherpa/collect/raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URI = "mongodb://sherpa_admin:sherpa2026!@localhost:27017"
DB_NAME = "sherpa_mongo"
COLLECTION = "bizinfo_raw"

def strip_cdata(text):
    if text is None:
        return None
    return re.sub(r'<[^>]+>', '', text).strip()

def parse_item(item):
    def get(tag):
        el = item.find(tag)
        return strip_cdata(el.text) if el is not None else None

    return {
        "seq": get("seq"),
        "title": get("title"),
        "link": get("link"),
        "author": get("author"),
        "excInsttNm": get("excInsttNm"),
        "description": get("description"),
        "lcategory": get("lcategory"),
        "pubDate": get("pubDate"),
        "reqstDt": get("reqstDt"),
        "trgetNm": get("trgetNm"),
    }

def fetch_bizinfo():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    all_items = []
    page = 1
    page_size = 100

    while True:
        params = {
            "crtfcKey": BIZINFO_API_KEY,
            "pageIndex": page,
            "pageUnit": page_size,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            print(f"[ERROR] page {page}: {e}")
            break

        items = root.findall(".//item")
        if not items:
            print(f"[DONE] 총 {len(all_items)}건 수집 완료")
            break

        parsed = [parse_item(item) for item in items]
        all_items.extend(parsed)
        print(f"[OK] page {page} → {len(parsed)}건 (누적: {len(all_items)}건)")

        for item in parsed:
            col.update_one(
                {"seq": item["seq"]},
                {"$set": item},
                upsert=True
            )

        if len(parsed) < page_size:
            print(f"[DONE] 마지막 페이지. 총 {len(all_items)}건")
            break

        page += 1
        time.sleep(0.3)

    today = datetime.now().strftime("%Y%m%d")
    out_path = SAVE_DIR / f"bizinfo_{today}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {out_path} ({len(all_items)}건)")

    client.close()
    return len(all_items)

if __name__ == "__main__":
    fetch_bizinfo()

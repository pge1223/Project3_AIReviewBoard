import requests
import json
import time
from datetime import datetime
from pymongo import MongoClient
from pathlib import Path

API_KEY = "b115bbf44b0b8c656ba172cf934837eeb818f28ec3fd3f8d0ce574496d9104a6"
BASE_URL = "https://apis.data.go.kr/B552735/kisedKstartupService01/getAnnouncementInformation01"
SAVE_DIR = Path("/mnt/hdd/sherpa/collect/raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URI = "mongodb://sherpa_admin:sherpa2026!@localhost:27017"
DB_NAME = "sherpa_mongo"
COLLECTION = "kstartup_raw"

def fetch_kstartup():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    all_items = []
    page = 1
    total_count = None

    while True:
        params = {
            "serviceKey": API_KEY,
            "pageNo": page,
            "numOfRows": 10,
            "returnType": "json",
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] page {page}: {e}")
            break

        if total_count is None:
            total_count = data.get("totalCount", 0)
            print(f"[INFO] 전체 건수: {total_count}건")

        items = data.get("data", [])
        if not items:
            print(f"[DONE] 총 {len(all_items)}건 수집 완료")
            break

        all_items.extend(items)
        print(f"[OK] page {page} → {len(items)}건 (누적: {len(all_items)}/{total_count}건)")

        for item in items:
            col.update_one(
                {"pbanc_sn": item.get("pbanc_sn")},
                {"$set": item},
                upsert=True
            )

        if len(all_items) >= total_count:
            print(f"[DONE] 수집 완료. 총 {len(all_items)}건")
            break

        page += 1
        time.sleep(0.3)

    today = datetime.now().strftime("%Y%m%d")
    out_path = SAVE_DIR / f"kstartup_{today}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {out_path} ({len(all_items)}건)")

    client.close()
    return len(all_items)

if __name__ == "__main__":
    fetch_kstartup()

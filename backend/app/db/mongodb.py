from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB]
    print(f"MongoDB 연결 완료: {settings.MONGODB_DB}")


async def close_db():
    global client
    if client:
        client.close()
        print("MongoDB 연결 종료")


def get_db():
    return db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.mongodb import connect_db, close_db

app = FastAPI(
    title="AI Review Board API",
    description="RAG 기반 AI 심사위원회 시스템",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await connect_db()


@app.on_event("shutdown")
async def shutdown():
    await close_db()


@app.get("/")
async def root():
    return {"message": "AI Review Board API is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
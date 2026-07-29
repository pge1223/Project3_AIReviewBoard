from typing import Optional
from fastapi import APIRouter, Header
from app.schemas.user import UserProfile, UserProfileResponse
from app.repositories.user_repository import UserRepository
from app.api.routes.documents import get_current_user

router = APIRouter(prefix="/users/me/profile", tags=["profile"])
user_repo = UserRepository()


@router.get("", response_model=UserProfileResponse)
async def get_my_profile(authorization: Optional[str] = Header(None, alias="authorization")):
    user_email = get_current_user(authorization)
    user = await user_repo.find_by_email(user_email)
    profile = (user or {}).get("profile") or {}
    return UserProfileResponse(**profile)


@router.put("", response_model=UserProfileResponse)
async def update_my_profile(
    request: UserProfile,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    dump = request.model_dump()
    updated = await user_repo.upsert_profile(user_email, dump)
    return UserProfileResponse(**updated["profile"])

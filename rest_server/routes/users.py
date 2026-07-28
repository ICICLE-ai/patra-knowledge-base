"""User self-registration (open -- no auth required)."""

from fastapi import APIRouter, Depends, HTTPException
import asyncpg

from rest_server.database import get_pool
from rest_server.models import UserRegistration

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserRegistration)
async def register_user(body: UserRegistration, pool: asyncpg.Pool = Depends(get_pool)):
    """Register a username (idempotent)."""
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users (username) VALUES ($1) ON CONFLICT (username) DO NOTHING",
            username,
        )
    return UserRegistration(username=username)

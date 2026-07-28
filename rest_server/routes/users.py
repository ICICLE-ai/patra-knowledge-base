"""User self-registration."""

from fastapi import APIRouter, Depends, HTTPException
import asyncpg

from rest_server.database import get_pool
from rest_server.deps import require_authenticated_actor, PatraActor
from rest_server.models import UserRegistration

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserRegistration)
async def register_user(
    pool: asyncpg.Pool = Depends(get_pool),
    actor: PatraActor = Depends(require_authenticated_actor),
):
    """Register the caller's own username (idempotent) so CKN ingest will accept events for it."""
    if not actor.username:
        raise HTTPException(status_code=400, detail="X-Patra-Username header required")
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users (username) VALUES ($1) ON CONFLICT (username) DO NOTHING",
            actor.username,
        )
    return UserRegistration(username=actor.username)

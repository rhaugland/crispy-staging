from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Verify JWT token and return agent/org info.
    Placeholder — wire up real JWT verification."""
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    # TODO: Real JWT verification
    return {"agent_id": "agent-1", "org_id": "org-1"}

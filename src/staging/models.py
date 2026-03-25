# src/staging/models.py
from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "pending"
    RECONSTRUCTING = "reconstructing"
    ANALYZING = "analyzing"
    LAYING_OUT = "laying_out"
    RENDERING = "rendering"
    HARMONIZING = "harmonizing"
    COMPLETED = "completed"
    FAILED = "failed"


class RoomType(StrEnum):
    LIVING = "living"
    BEDROOM = "bedroom"
    DINING = "dining"
    OFFICE = "office"


class Style(StrEnum):
    MODERN = "modern"
    TRADITIONAL = "traditional"


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    org_id: str
    photo_keys: list[str]
    room_type: RoomType | None = None
    style: Style = Style.MODERN
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    result_keys: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class CameraPose(BaseModel):
    position: Any  # np.ndarray (3,)
    rotation: Any  # np.ndarray (3, 3)
    focal_length: float
    image_width: int
    image_height: int

    model_config = {"arbitrary_types_allowed": True}


class FloorPlan(BaseModel):
    polygon: Any  # np.ndarray (N, 2)
    plane_normal: Any  # np.ndarray (3,)
    plane_offset: float

    model_config = {"arbitrary_types_allowed": True}

    def area(self) -> float:
        """Shoelace formula for polygon area."""
        pts = self.polygon
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0


class ExclusionZone(BaseModel):
    label: str
    polygon: Any  # np.ndarray (N, 2) on floor plane
    model_config = {"arbitrary_types_allowed": True}


class FurniturePlacement(BaseModel):
    asset_id: str
    position: Any  # np.ndarray (3,) world coords
    rotation_y: float  # degrees, rotation around vertical axis
    scale: float = 1.0
    model_config = {"arbitrary_types_allowed": True}

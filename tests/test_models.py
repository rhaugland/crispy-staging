# tests/test_models.py
import pytest
import numpy as np
from staging.models import (
    Job, JobStatus, CameraPose, FloorPlan,
    ExclusionZone, FurniturePlacement, RoomType, Style,
)


def test_job_creation():
    job = Job(agent_id="agent-1", org_id="org-1", photo_keys=["a.jpg", "b.jpg", "c.jpg"])
    assert job.status == JobStatus.PENDING
    assert job.id is not None
    assert len(job.photo_keys) == 3


def test_camera_pose_matrix():
    pose = CameraPose(
        position=np.array([1.0, 2.0, 3.0]),
        rotation=np.eye(3),
        focal_length=35.0,
        image_width=1920,
        image_height=1080,
    )
    assert pose.position.shape == (3,)
    assert pose.rotation.shape == (3, 3)


def test_floor_plan_area():
    plan = FloorPlan(
        polygon=np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float),
        plane_normal=np.array([0, 1, 0], dtype=float),
        plane_offset=0.0,
    )
    assert plan.area() == pytest.approx(12.0)


def test_room_type_enum():
    assert RoomType.LIVING == "living"
    assert RoomType.BEDROOM == "bedroom"
    assert RoomType.DINING == "dining"
    assert RoomType.OFFICE == "office"


def test_style_enum():
    assert Style.MODERN == "modern"
    assert Style.TRADITIONAL == "traditional"

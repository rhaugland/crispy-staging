import numpy as np
import pytest
from staging.layout.placer import FurniturePlacer
from staging.layout.assets import FurnitureAsset
from staging.models import FloorPlan, ExclusionZone, RoomType, Style, FurniturePlacement


def _make_floor(w=5.0, d=4.0):
    return FloorPlan(
        polygon=np.array([[0, 0], [w, 0], [w, d], [0, d]], dtype=float),
        plane_normal=np.array([0, 1, 0], dtype=float),
        plane_offset=0.0,
    )


def _make_asset(asset_id, category, w=1.0, d=0.5, h=0.8):
    return FurnitureAsset(
        id=asset_id, name=asset_id, category=category,
        room_types=["living"], styles=["modern"],
        dimensions={"width": w, "depth": d, "height": h},
        file=f"{asset_id}.glb",
    )


def test_place_furniture_in_room():
    placer = FurniturePlacer()
    floor = _make_floor()
    assets = [_make_asset("sofa-1", "sofa", w=2.0, d=0.9), _make_asset("table-1", "coffee_table", w=1.2, d=0.6)]
    placements = placer.place(
        room_type=RoomType.LIVING,
        floor=floor,
        available_assets=assets,
        exclusion_zones=[],
    )
    assert len(placements) >= 2
    for p in placements:
        assert isinstance(p, FurniturePlacement)


def test_placement_avoids_exclusion_zones():
    placer = FurniturePlacer()
    floor = _make_floor(w=5.0, d=4.0)
    exclusion = ExclusionZone(
        label="door",
        polygon=np.array([[2, 3.5], [3, 3.5], [3, 4], [2, 4]], dtype=float),
    )
    assets = [_make_asset("sofa-1", "sofa", w=2.0, d=0.9)]
    placements = placer.place(
        room_type=RoomType.LIVING,
        floor=floor,
        available_assets=assets,
        exclusion_zones=[exclusion],
    )
    for p in placements:
        pos = p.position
        assert not (2.0 <= pos[0] <= 3.0 and 3.5 <= pos[2] <= 4.0)


def test_small_room_reduces_furniture():
    placer = FurniturePlacer()
    floor = _make_floor(w=2.0, d=1.5)
    assets = [_make_asset("sofa-1", "sofa", w=2.0, d=0.9)]
    placements = placer.place(
        room_type=RoomType.LIVING,
        floor=floor,
        available_assets=assets,
        exclusion_zones=[],
    )
    assert len(placements) >= 1

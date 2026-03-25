from __future__ import annotations
import numpy as np
from staging.layout.assets import FurnitureAsset
from staging.layout.templates import get_template, TemplateSlot
from staging.models import ExclusionZone, FloorPlan, FurniturePlacement, RoomType

PLACEABLE_RATIO_REDUCED = 0.30
PLACEABLE_RATIO_HERO_ONLY = 0.15


class FurniturePlacer:
    def place(
        self,
        room_type: RoomType,
        floor: FloorPlan,
        available_assets: list[FurnitureAsset],
        exclusion_zones: list[ExclusionZone],
    ) -> list[FurniturePlacement]:
        template = get_template(room_type)
        bbox = self._bounding_box(floor.polygon)
        room_width = bbox[2] - bbox[0]
        room_depth = bbox[3] - bbox[1]
        origin_x, origin_z = bbox[0], bbox[1]

        placeable_ratio = self._placeable_ratio(floor, exclusion_zones)
        if placeable_ratio < PLACEABLE_RATIO_HERO_ONLY:
            slots = [s for s in template.slots if s.required][:1]
        elif placeable_ratio < PLACEABLE_RATIO_REDUCED:
            slots = [s for s in template.slots if s.required]
        else:
            slots = template.slots

        placements: list[FurniturePlacement] = []
        for slot in slots:
            asset = self._find_asset(available_assets, slot.category)
            if asset is None:
                continue

            world_x = origin_x + slot.relative_x * room_width
            world_z = origin_z + slot.relative_z * room_depth
            position = np.array([world_x, 0.0, world_z])

            if self._collides_with_exclusion(position, asset, exclusion_zones):
                if not slot.required:
                    continue
                position = self._nudge_away(position, asset, exclusion_zones, floor)

            placements.append(FurniturePlacement(
                asset_id=asset.id,
                position=position,
                rotation_y=slot.rotation_y,
            ))

        return placements

    def _find_asset(self, assets: list[FurnitureAsset], category: str) -> FurnitureAsset | None:
        matches = [a for a in assets if a.category == category]
        return matches[0] if matches else None

    def _bounding_box(self, polygon: np.ndarray) -> tuple[float, float, float, float]:
        return (polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max())

    def _placeable_ratio(self, floor: FloorPlan, zones: list[ExclusionZone]) -> float:
        from staging.analysis.fixtures import compute_placeable_ratio
        return compute_placeable_ratio(floor, zones)

    def _collides_with_exclusion(
        self, pos: np.ndarray, asset: FurnitureAsset, zones: list[ExclusionZone]
    ) -> bool:
        hw = asset.dimensions["width"] / 2
        hd = asset.dimensions["depth"] / 2
        for zone in zones:
            zmin = zone.polygon.min(axis=0)
            zmax = zone.polygon.max(axis=0)
            if (pos[0] + hw > zmin[0] and pos[0] - hw < zmax[0] and
                    pos[2] + hd > zmin[1] and pos[2] - hd < zmax[1]):
                return True
        return False

    def _nudge_away(
        self, pos: np.ndarray, asset: FurnitureAsset,
        zones: list[ExclusionZone], floor: FloorPlan,
    ) -> np.ndarray:
        for dx, dz in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (0.5, 0.5), (-0.5, -0.5)]:
            candidate = pos.copy()
            candidate[0] += dx
            candidate[2] += dz
            if not self._collides_with_exclusion(candidate, asset, zones):
                return candidate
        return pos

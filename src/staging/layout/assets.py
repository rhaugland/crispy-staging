from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from staging.models import RoomType, Style


@dataclass
class FurnitureAsset:
    id: str
    name: str
    category: str
    room_types: list[str]
    styles: list[str]
    dimensions: dict[str, float]
    file: str


class AssetCatalog:
    def __init__(self, manifest_path: Path):
        with open(manifest_path) as f:
            data = json.load(f)
        self._assets = [FurnitureAsset(**a) for a in data["assets"]]

    def filter(self, room_type: RoomType, style: Style) -> list[FurnitureAsset]:
        return [
            a for a in self._assets
            if room_type.value in a.room_types and style.value in a.styles
        ]

    def get_by_category(
        self, room_type: RoomType, style: Style, category: str
    ) -> list[FurnitureAsset]:
        return [a for a in self.filter(room_type, style) if a.category == category]

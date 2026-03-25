import json
import pytest
from pathlib import Path
from staging.layout.assets import AssetCatalog
from staging.models import RoomType, Style


@pytest.fixture
def catalog(tmp_path):
    manifest = {
        "assets": [
            {"id": "modern-sofa-01", "name": "Modern Sofa", "category": "sofa", "room_types": ["living"], "styles": ["modern"], "dimensions": {"width": 2.2, "depth": 0.9, "height": 0.85}, "file": "modern-sofa-01.glb"},
            {"id": "modern-coffee-table-01", "name": "Modern Coffee Table", "category": "coffee_table", "room_types": ["living"], "styles": ["modern"], "dimensions": {"width": 1.2, "depth": 0.6, "height": 0.45}, "file": "modern-coffee-table-01.glb"},
            {"id": "trad-sofa-01", "name": "Traditional Sofa", "category": "sofa", "room_types": ["living"], "styles": ["traditional"], "dimensions": {"width": 2.0, "depth": 0.95, "height": 0.9}, "file": "trad-sofa-01.glb"},
            {"id": "modern-bed-01", "name": "Modern Bed", "category": "bed", "room_types": ["bedroom"], "styles": ["modern"], "dimensions": {"width": 1.6, "depth": 2.1, "height": 0.5}, "file": "modern-bed-01.glb"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return AssetCatalog(manifest_path=manifest_path)


def test_filter_by_room_and_style(catalog):
    assets = catalog.filter(room_type=RoomType.LIVING, style=Style.MODERN)
    assert len(assets) == 2
    assert all(a.id.startswith("modern") for a in assets)


def test_filter_bedroom(catalog):
    assets = catalog.filter(room_type=RoomType.BEDROOM, style=Style.MODERN)
    assert len(assets) == 1
    assert assets[0].category == "bed"


def test_get_by_category(catalog):
    sofas = catalog.get_by_category(room_type=RoomType.LIVING, style=Style.MODERN, category="sofa")
    assert len(sofas) == 1
    assert sofas[0].id == "modern-sofa-01"


def test_asset_dimensions(catalog):
    assets = catalog.filter(room_type=RoomType.LIVING, style=Style.MODERN)
    sofa = [a for a in assets if a.category == "sofa"][0]
    assert sofa.dimensions["width"] == 2.2

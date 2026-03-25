import pytest
from staging.layout.templates import get_template, LayoutTemplate
from staging.models import RoomType


def test_living_room_template():
    template = get_template(RoomType.LIVING)
    assert template is not None
    assert "sofa" in [slot.category for slot in template.slots]
    assert "coffee_table" in [slot.category for slot in template.slots]


def test_bedroom_template():
    template = get_template(RoomType.BEDROOM)
    assert "bed" in [slot.category for slot in template.slots]
    assert "nightstand" in [slot.category for slot in template.slots]


def test_all_room_types_have_templates():
    for room_type in RoomType:
        template = get_template(room_type)
        assert template is not None
        assert len(template.slots) >= 2

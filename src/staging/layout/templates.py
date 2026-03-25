from __future__ import annotations
from dataclasses import dataclass
from staging.models import RoomType


@dataclass
class TemplateSlot:
    category: str
    relative_x: float
    relative_z: float
    rotation_y: float
    required: bool = True


@dataclass
class LayoutTemplate:
    room_type: RoomType
    slots: list[TemplateSlot]


TEMPLATES: dict[RoomType, LayoutTemplate] = {
    RoomType.LIVING: LayoutTemplate(
        room_type=RoomType.LIVING,
        slots=[
            TemplateSlot(category="sofa", relative_x=0.5, relative_z=0.65, rotation_y=0),
            TemplateSlot(category="coffee_table", relative_x=0.5, relative_z=0.45, rotation_y=0),
            TemplateSlot(category="accent_chair", relative_x=0.2, relative_z=0.45, rotation_y=90, required=False),
            TemplateSlot(category="rug", relative_x=0.5, relative_z=0.5, rotation_y=0, required=False),
        ],
    ),
    RoomType.BEDROOM: LayoutTemplate(
        room_type=RoomType.BEDROOM,
        slots=[
            TemplateSlot(category="bed", relative_x=0.5, relative_z=0.6, rotation_y=0),
            TemplateSlot(category="nightstand", relative_x=0.15, relative_z=0.6, rotation_y=0),
            TemplateSlot(category="nightstand", relative_x=0.85, relative_z=0.6, rotation_y=0),
            TemplateSlot(category="dresser", relative_x=0.5, relative_z=0.1, rotation_y=180, required=False),
        ],
    ),
    RoomType.DINING: LayoutTemplate(
        room_type=RoomType.DINING,
        slots=[
            TemplateSlot(category="dining_table", relative_x=0.5, relative_z=0.5, rotation_y=0),
            TemplateSlot(category="dining_chair", relative_x=0.3, relative_z=0.5, rotation_y=90),
            TemplateSlot(category="dining_chair", relative_x=0.7, relative_z=0.5, rotation_y=-90),
            TemplateSlot(category="dining_chair", relative_x=0.5, relative_z=0.35, rotation_y=0, required=False),
            TemplateSlot(category="dining_chair", relative_x=0.5, relative_z=0.65, rotation_y=180, required=False),
        ],
    ),
    RoomType.OFFICE: LayoutTemplate(
        room_type=RoomType.OFFICE,
        slots=[
            TemplateSlot(category="desk", relative_x=0.5, relative_z=0.8, rotation_y=0),
            TemplateSlot(category="office_chair", relative_x=0.5, relative_z=0.6, rotation_y=0),
            TemplateSlot(category="bookshelf", relative_x=0.1, relative_z=0.5, rotation_y=90, required=False),
        ],
    ),
}


def get_template(room_type: RoomType) -> LayoutTemplate:
    return TEMPLATES[room_type]

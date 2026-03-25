import numpy as np
import pytest
from unittest.mock import patch, AsyncMock
from staging.analysis.classifier import RoomClassifier
from staging.models import RoomType


@pytest.mark.asyncio
async def test_classify_returns_room_type():
    classifier = RoomClassifier(api_key="test-key")
    with patch.object(classifier, "_call_vision_model", new_callable=AsyncMock, return_value="living"):
        result = await classifier.classify(images=[np.zeros((1080, 1920, 3), dtype=np.uint8)])
    assert result == RoomType.LIVING


@pytest.mark.asyncio
async def test_classify_bedroom():
    classifier = RoomClassifier(api_key="test-key")
    with patch.object(classifier, "_call_vision_model", new_callable=AsyncMock, return_value="bedroom"):
        result = await classifier.classify(images=[np.zeros((1080, 1920, 3), dtype=np.uint8)])
    assert result == RoomType.BEDROOM


@pytest.mark.asyncio
async def test_classify_unknown_defaults_to_living():
    classifier = RoomClassifier(api_key="test-key")
    with patch.object(classifier, "_call_vision_model", new_callable=AsyncMock, return_value="sunroom"):
        result = await classifier.classify(images=[np.zeros((1080, 1920, 3), dtype=np.uint8)])
    assert result == RoomType.LIVING

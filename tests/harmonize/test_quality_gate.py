import numpy as np
import pytest
from unittest.mock import patch, AsyncMock
from staging.harmonize.quality_gate import QualityGate, QualityCheckResult


@pytest.mark.asyncio
async def test_quality_passes():
    gate = QualityGate(api_key="test-key")
    images = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    originals = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    with patch.object(gate, "_check_with_vision", new_callable=AsyncMock, return_value=QualityCheckResult(passed=True, issues=[], failed_indices=[])):
        result = await gate.check(staged_images=images, original_images=originals)
    assert result.passed is True


@pytest.mark.asyncio
async def test_quality_fails_with_issues():
    gate = QualityGate(api_key="test-key")
    images = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    originals = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    fail_result = QualityCheckResult(
        passed=False,
        issues=["Wall appears modified in image 2"],
        failed_indices=[1],
    )
    with patch.object(gate, "_check_with_vision", new_callable=AsyncMock, return_value=fail_result):
        result = await gate.check(staged_images=images, original_images=originals)
    assert result.passed is False
    assert len(result.failed_indices) == 1


def test_retry_params_adjustment():
    gate = QualityGate(api_key="test-key")
    params = gate._adjust_params(attempt=0)
    assert params["color_temp_shift"] == 0
    params = gate._adjust_params(attempt=1)
    assert params["color_temp_shift"] != 0

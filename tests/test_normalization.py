import pytest
import torch
from pathlib import Path
from transformer_lib.normalization import RMSNorm

FIXTURES_PATH = Path(__file__).parent / "fixtures"



def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: dict[str, torch.FloatTensor],
        in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    
    weight = weights['weight']

    # 创建RMSNorm实例
    rms_norm_layer = RMSNorm(d_model=d_model, epsilon=eps)

    # 将权重设置到RMSNorm层
    rms_norm_layer.weight.data = weight

    # 调用RMSNorm的forward方法
    output = rms_norm_layer(in_features)

    return output


def test_rmsnorm():
    reference_weights = torch.load(FIXTURES_PATH / "rmsnorm_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "rmsnorm_expected_output.pt")
    d_model = 64
    actual_output = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_features
    )

    tolerance = 1e-6
    assert torch.allclose(actual_output, expected_output, atol=tolerance), "Outputs are not close enough!"
import torch

from dsm.dsm_metric import compute_dsm


def test_dsm_squared_l2_differences():
    g0 = torch.tensor([1.0, 0.0, 2.0])
    g1 = torch.tensor([2.0, 0.0, 0.0])
    g2 = torch.tensor([2.0, 3.0, 0.0])

    out = compute_dsm([g0, g1, g2])

    assert out["dsm_0_1"] == 5.0
    assert out["dsm_1_2"] == 9.0

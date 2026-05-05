import torch

from dsm.score_utils import model_output_to_epsilon, model_output_to_score


def test_epsilon_prediction_conversion():
    model_output = torch.tensor([1.0, -2.0])
    sample = torch.tensor([0.5, 0.5])
    alpha = torch.tensor(0.8)
    sigma = torch.tensor(0.6)

    eps = model_output_to_epsilon(model_output, sample, alpha, sigma, "epsilon")
    score = model_output_to_score(model_output, sample, alpha, sigma, "epsilon")

    assert torch.equal(eps, model_output)
    assert torch.allclose(score, -model_output / sigma)


def test_v_prediction_conversion_shape_dtype():
    v = torch.tensor([1.0, -2.0], dtype=torch.float32)
    sample = torch.tensor([0.5, 0.25], dtype=torch.float32)
    alpha = torch.tensor(0.8, dtype=torch.float32)
    sigma = torch.tensor(0.6, dtype=torch.float32)

    eps = model_output_to_epsilon(v, sample, alpha, sigma, "v_prediction")

    assert eps.shape == v.shape
    assert eps.dtype == v.dtype
    assert torch.allclose(eps, alpha * v + sigma * sample)

import sys
import types

import torch


def test_load_pipeline_uses_local_files_only_by_default(monkeypatch):
    from dsm.diffusion import load_pipeline

    calls = {}

    class DummyScheduler:
        config = {}

        @classmethod
        def from_config(cls, config):
            return cls()

    class DummyPipeline:
        scheduler = DummyScheduler()

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["device"] = device

        def set_progress_bar_config(self, disable):
            calls["progress_disabled"] = disable

    fake_diffusers = types.SimpleNamespace(
        DDIMScheduler=DummyScheduler,
        StableDiffusionPipeline=DummyPipeline,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    load_pipeline("checkpoints/sd-v1-4", torch.float16, "cpu")

    assert calls["model_name"] == "checkpoints/sd-v1-4"
    assert calls["kwargs"]["local_files_only"] is True
    assert calls["kwargs"]["torch_dtype"] is torch.float16


def test_load_pipeline_allow_download_opt_in(monkeypatch):
    from dsm.diffusion import load_pipeline

    calls = {}

    class DummyScheduler:
        config = {}

        @classmethod
        def from_config(cls, config):
            return cls()

    class DummyPipeline:
        scheduler = DummyScheduler()

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["kwargs"] = kwargs
            return cls()

        def to(self, device):
            pass

        def set_progress_bar_config(self, disable):
            pass

    fake_diffusers = types.SimpleNamespace(
        DDIMScheduler=DummyScheduler,
        StableDiffusionPipeline=DummyPipeline,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    load_pipeline("CompVis/stable-diffusion-v1-4", torch.float16, "cpu", allow_download=True)

    assert calls["kwargs"]["local_files_only"] is False

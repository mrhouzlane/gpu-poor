from types import SimpleNamespace

import pytest

from gpu_poor import models
from gpu_poor.models import load_model, parse_config


def test_hand_checked_llama_8b(llama):
    # 2 (K,V) * 32 layers * 8 KV heads * 128 dim * 2 BF16 bytes.
    assert llama.kv_bytes_per_token() == 131_072
    assert llama.kv_bytes_per_token("fp8") == 65_536
    assert llama.parameters == 8_030_261_248


def test_gqa_vs_mha(llama_config, llama):
    llama_config.pop("num_key_value_heads")
    mha = parse_config(llama_config)
    assert mha.kv_heads == 32
    assert mha.kv_bytes_per_token() == 4 * llama.kv_bytes_per_token()


def test_tied_embeddings(llama_config, llama):
    llama_config["tie_word_embeddings"] = True
    assert parse_config(llama_config).parameters == llama.parameters - 128256 * 4096


def test_qwen_biases(llama_config, llama):
    llama_config["model_type"] = "qwen2"
    qwen = parse_config(llama_config)
    assert qwen.parameters == llama.parameters + 32 * (4096 + 2 * 1024)


def test_attention_and_mlp_bias(llama_config, llama):
    llama_config.update(attention_bias=True, mlp_bias=True)
    assert parse_config(llama_config).parameters == llama.parameters + 32 * (
        4096 + 2 * 1024 + 4096 + 2 * 14336 + 4096
    )


def test_explicit_head_dim(llama_config):
    llama_config["head_dim"] = 64
    assert parse_config(llama_config).head_dim == 64


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "max_position_embeddings",
    ],
)
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5])
def test_invalid_dimensions(llama_config, field, value):
    llama_config[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        parse_config(llama_config)


def test_missing_required_fields_not_registry_defaults(llama_config):
    del llama_config["num_hidden_layers"]
    with pytest.raises(ValueError):
        parse_config(llama_config)


@pytest.mark.parametrize("kind", ["mixtral", "deepseek_v3", "t5", "bert", "mamba", None])
def test_unsupported_architectures(llama_config, kind):
    llama_config["model_type"] = kind
    with pytest.raises(ValueError, match="Unsupported"):
        parse_config(llama_config)


def test_quantized_config_rejected(llama_config):
    llama_config["quantization_config"] = {"quant_method": "gptq"}
    with pytest.raises(ValueError, match="original dense"):
        parse_config(llama_config)


def test_nested_and_expert_rejected(llama_config):
    for field in ["text_config", "num_local_experts", "num_experts"]:
        with pytest.raises(ValueError, match="not supported"):
            parse_config({**llama_config, field: {"a": 1}})


def test_bad_grouping(llama_config):
    llama_config["num_key_value_heads"] = 7
    with pytest.raises(ValueError, match="divisible"):
        parse_config(llama_config)


def test_bad_head_dim_without_explicit(llama_config):
    llama_config.pop("head_dim")
    llama_config["hidden_size"] = 4097
    with pytest.raises(ValueError, match="divide"):
        parse_config(llama_config)


def test_kv_dtype_rejects_int4(llama):
    with pytest.raises(ValueError, match="KV dtype"):
        llama.kv_bytes_per_token("int4")


def test_sliding_window_recorded_not_silently_applied(llama_config):
    llama_config.update(model_type="mistral", sliding_window=4096)
    assert parse_config(llama_config).sliding_window == 4096
    llama_config.update(model_type="qwen2", use_sliding_window=False)
    assert parse_config(llama_config).sliding_window is None


def test_hub_download_config_only(monkeypatch, tmp_path, llama_config):
    path = tmp_path / "snapshots" / "abc123" / "config.json"
    path.parent.mkdir(parents=True)
    import json

    path.write_text(json.dumps(llama_config))
    calls = []

    def download(model_id, filename, **kwargs):
        calls.append((model_id, filename, kwargs))
        return str(path)

    monkeypatch.setattr(models, "hf_hub_download", download)
    spec = load_model("test/model", local_files_only=True)
    assert calls[0][1] == "config.json"
    assert calls[0][2]["local_files_only"] is True
    assert spec.revision == "abc123"


def test_safetensors_metadata(monkeypatch, tmp_path, llama_config):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(llama_config))
    monkeypatch.setattr(models, "hf_hub_download", lambda *a, **kw: str(path))
    called = []

    def meta(self, model_id, **kwargs):
        called.append((model_id, kwargs))
        return SimpleNamespace(parameter_count={"BF16": 100, "F32": 20})

    monkeypatch.setattr(models.HfApi, "get_safetensors_metadata", meta)
    spec = load_model("test/model", parameter_source="safetensors")
    assert spec.parameters == 120
    assert spec.parameter_source == "safetensors headers"
    assert len(called) == 1


def test_offline_local_config():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures/llama3-8b.json"
    assert load_model("fixture", config_path=path).parameters == 8_030_261_248
    with pytest.raises(ValueError, match="needs Hub"):
        load_model("fixture", config_path=path, parameter_source="safetensors")


def test_malformed_config():
    with pytest.raises(ValueError, match="object"):
        parse_config([])
    with pytest.raises(ValueError, match="Unsupported"):
        parse_config({"model_type": []})


def test_metadata_absent_method_falls_back_with_warning(monkeypatch, tmp_path, llama_config):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(llama_config))
    monkeypatch.setattr(models, "hf_hub_download", lambda *a, **kw: str(path))
    monkeypatch.delattr(models.HfApi, "get_safetensors_metadata")
    with pytest.warns(UserWarning, match="lacks safetensors"):
        spec = load_model("fixture", parameter_source="safetensors")
    assert spec.parameter_source == "config formula"


def test_empty_metadata_rejected(monkeypatch, tmp_path, llama_config):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(llama_config))
    monkeypatch.setattr(models, "hf_hub_download", lambda *a, **kw: str(path))
    monkeypatch.setattr(
        models.HfApi,
        "get_safetensors_metadata",
        lambda *a, **kw: SimpleNamespace(parameter_count={}),
    )
    with pytest.raises(ValueError, match="Empty safetensors"):
        load_model("fixture", parameter_source="safetensors")


def test_parameter_source_validation():
    with pytest.raises(ValueError, match="parameter_source"):
        load_model("fixture", parameter_source="weights")

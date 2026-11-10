import json
from pathlib import Path

import pytest

from gpu_poor import cli

FIXTURE = str(Path(__file__).parent / "fixtures/llama3-8b.json")


def test_cli_json_offline(capsys):
    result = cli.main(["fixture", "--config", FIXTURE, "--json", "--batch", "1,8"])
    assert result == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"]["parameters"] == 8_030_261_248
    assert len(data["estimates"]) == 2
    assert "THEORETICAL" in data["notice"]
    assert data["prices"]["is_example"] is False


def test_cli_text(capsys):
    assert cli.main(["fixture", "--config", FIXTURE, "--batch", "1,32"]) == 0
    output = capsys.readouterr().out
    assert "$/M out" in output
    assert "131,072 KV bytes/token" in output
    assert "archived research data" in output


def test_cli_no_prices_and_oom(capsys):
    assert (
        cli.main(
            ["fixture", "--config", FIXTURE, "--no-prices", "--gpu", "RTX4090", "--batch", "32"]
        )
        == 0
    )
    assert "NO" in capsys.readouterr().out


def test_cli_error(capsys):
    assert cli.main(["fixture", "--config", FIXTURE, "--context", "-1"]) == 2
    assert "positive" in capsys.readouterr().err


@pytest.mark.parametrize("batches", ["0", "-1", "x", "1,,8"])
def test_bad_batch(batches):
    with pytest.raises(SystemExit):
        cli.main(["fixture", "--batch", batches])


def test_app_function_uses_library(monkeypatch, llama):
    import app

    monkeypatch.setattr(app, "load_model", lambda *a: llama)
    text, data = app.estimate_ui("fixture", "H100", "bf16", "1,8", 4096, 0.9, None, True)
    assert "Archived research" in text
    assert data["model"]["parameters"] == 8_030_261_248
    assert len(data["estimates"]) == 2


def test_app_invalid_batch_is_readable_error(monkeypatch, llama):
    import gradio as gr

    import app

    monkeypatch.setattr(app, "load_model", lambda *a: llama)
    with pytest.raises(gr.Error, match="positive integers"):
        app.estimate_ui("fixture", "H100", "bf16", "0", 4096, 0.9, None, True)


def test_hub_errors_have_no_traceback(monkeypatch, capsys):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    def fail(*args, **kwargs):
        raise HfHubHTTPError(
            "public config unavailable",
            response=httpx.Response(404, request=httpx.Request("GET", "https://example.org")),
        )

    monkeypatch.setattr(cli, "load_model", fail)
    assert cli.main(["missing/model"]) == 2
    assert "public config unavailable" in capsys.readouterr().err

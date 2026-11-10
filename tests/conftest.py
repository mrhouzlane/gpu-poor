"""Tests are offline: any accidental external HTTP request fails immediately."""

import json
import socket
from pathlib import Path

import httpx
import pytest

from gpu_poor.models import parse_config


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Network is forbidden in unit tests")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(httpx.Client, "request", deny)
    monkeypatch.setattr(httpx.AsyncClient, "request", deny)


@pytest.fixture
def llama_config():
    return json.loads((Path(__file__).parent / "fixtures/llama3-8b.json").read_text())


@pytest.fixture
def llama(llama_config):
    return parse_config(llama_config, "fixture/llama3-8b")

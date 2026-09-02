import os
from pathlib import Path
from types import SimpleNamespace

import pytest


os.environ["DOTENV_PATH"] = str(
    Path(__file__).parent / "dotenv_data" / "dotenv_no_datasource_1"
)
os.environ["MS_DEFENDER_ENABLED"] = "false"

import app as app_module
from backend.azure_search import RetrievalResult


def test_prepare_model_args_uses_gpt_5_1_contract():
    model_args = app_module.prepare_model_args(
        {"messages": [{"role": "user", "content": "hello"}]},
        {},
    )

    assert model_args["max_completion_tokens"] == 4000
    assert model_args["extra_body"] == {"reasoning_effort": "none"}
    assert "max_tokens" not in model_args
    assert "temperature" not in model_args
    assert "top_p" not in model_args


def test_prepare_model_args_applies_validated_user_preferences():
    model_args = app_module.prepare_model_args(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "user_settings": {
                "response_length": 2000,
                "reasoning_effort": "medium",
                "data_grounding": False,
                "retrieved_documents": 3,
                "show_citations": False,
            },
        },
        {},
    )

    assert model_args["max_completion_tokens"] == 2000
    assert model_args["extra_body"] == {"reasoning_effort": "medium"}


def test_normalize_user_settings_rejects_invalid_document_count():
    with pytest.raises(ValueError, match="retrieved_documents"):
        app_module.normalize_user_settings({"retrieved_documents": 11})


def test_normalize_user_settings_allows_extended_response_length():
    settings = app_module.normalize_user_settings({"response_length": 32000})

    assert settings["response_length"] == 32000


def test_prepare_model_args_preserves_legacy_contract(monkeypatch):
    monkeypatch.setattr(
        app_module.app_settings.azure_openai,
        "deployed_model_name",
        "gpt-4.1",
    )

    model_args = app_module.prepare_model_args(
        {"messages": [{"role": "user", "content": "hello"}]},
        {},
    )

    assert model_args["max_tokens"] == 4000
    assert model_args["temperature"] == 0
    assert model_args["top_p"] == 1.0
    assert "max_completion_tokens" not in model_args
    assert "extra_body" not in model_args


def test_prepare_model_args_adds_direct_search_grounding(monkeypatch):
    monkeypatch.setattr(
        app_module.app_settings.base_settings,
        "datasource_type",
        "AzureCognitiveSearch",
    )

    model_args = app_module.prepare_model_args(
        {"messages": [{"role": "user", "content": "What is covered?"}]},
        {},
        grounding_context="[doc1] Coverage details",
    )

    assert model_args["messages"][0]["role"] == "system"
    assert "[doc1] Coverage details" in model_args["messages"][0]["content"]
    assert "data_sources" not in model_args.get("extra_body", {})


@pytest.mark.asyncio
async def test_send_chat_request_retrieves_before_gpt_5_1_call(monkeypatch):
    captured = {}

    class FakeRawResponse:
        headers = {"apim-request-id": "request-id"}

        def parse(self):
            return "completion"

    class FakeCreate:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeRawResponse()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=FakeCreate(),
            )
        )
    )

    async def fake_init_openai_client():
        return fake_client

    retrieval_args = {}

    async def fake_retrieve(**kwargs):
        retrieval_args.update(kwargs)
        return RetrievalResult(
            context="[doc1] Retrieved coverage",
            citations=[{"id": "1", "content": "Retrieved coverage"}],
        )

    search_settings = SimpleNamespace(key="search-key")
    monkeypatch.setattr(
        app_module.app_settings.base_settings,
        "datasource_type",
        "AzureCognitiveSearch",
    )
    monkeypatch.setattr(app_module.app_settings, "datasource", search_settings)
    monkeypatch.setattr(app_module, "init_openai_client", fake_init_openai_client)
    monkeypatch.setattr(app_module, "retrieve_from_azure_search", fake_retrieve)

    response, request_id, citations = await app_module.send_chat_request(
        {"messages": [{"role": "user", "content": "What is covered?"}]},
        {},
    )

    assert response == "completion"
    assert request_id == "request-id"
    assert citations == [{"id": "1", "content": "Retrieved coverage"}]
    assert "[doc1] Retrieved coverage" in captured["messages"][0]["content"]
    assert "data_sources" not in captured.get("extra_body", {})
    assert retrieval_args["include_citations"] is True

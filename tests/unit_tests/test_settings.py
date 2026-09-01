import os
import pytest
from importlib import import_module, reload


@pytest.fixture(scope="function")
def dotenv_path(request):
    test_case_name = request.node.originalname.partition("test_")[2]
    return os.path.join(
        os.path.dirname(__file__),
        "dotenv_data",
        test_case_name
    )


@pytest.fixture(scope="function")
def app_settings(dotenv_path):
    # Reload module object to pick up new environment
    os.environ["DOTENV_PATH"] = dotenv_path
    settings_module = import_module("backend.settings")
    settings_module = reload(settings_module)
    
    yield getattr(settings_module, "app_settings")


def test_dotenv_no_datasource_1(app_settings):    
    # Validate model object
    assert app_settings.base_settings.datasource_type is None
    assert app_settings.datasource is None
    assert app_settings.azure_openai is not None
    assert app_settings.azure_openai.deployed_model_name == "gpt-5.1"
    assert app_settings.azure_openai.is_reasoning_model is True
    assert app_settings.azure_openai.get_chat_completion_parameters() == {
        "max_completion_tokens": 1000,
        "reasoning_effort": "none",
    }
    app_settings.azure_openai.deployed_model_name = "gpt-4.1"
    app_settings.azure_openai.model = "gpt-5.1"
    assert app_settings.azure_openai.is_reasoning_model is True
    assert app_settings.azure_openai.get_chat_completion_parameters() == {
        "max_completion_tokens": 1000,
        "reasoning_effort": "none",
    }
    
    
def test_dotenv_invalid_azure_search_rejected(dotenv_path, monkeypatch):
    monkeypatch.setenv("DOTENV_PATH", dotenv_path)
    settings_module = import_module("backend.settings")

    with pytest.raises(ValueError, match="Invalid Azure AI Search configuration"):
        reload(settings_module)

    
def test_dotenv_with_azure_search_success(app_settings):
    # Validate model object
    assert app_settings.search is not None
    assert app_settings.base_settings.datasource_type == "AzureCognitiveSearch"
    assert app_settings.datasource is not None
    assert app_settings.datasource.service == "search_service"
    assert app_settings.azure_openai is not None
    assert app_settings.azure_openai.is_reasoning_model is False
    assert app_settings.azure_openai.get_chat_completion_parameters() == {
        "temperature": 0,
        "max_tokens": 1000,
        "top_p": 1.0,
        "stop": None,
    }
    
    # Validate API payload structure
    payload = app_settings.datasource.construct_payload_configuration()
    assert payload["type"] == "azure_search"
    assert payload["parameters"] is not None
    assert payload["parameters"]["endpoint"] == "https://search_service.search.azure.us"
    print(payload)


def test_dotenv_with_elasticsearch_success(app_settings):
    # Validate model object
    assert app_settings.search is not None
    assert app_settings.base_settings.datasource_type == "Elasticsearch"
    assert app_settings.datasource is not None
    assert app_settings.datasource.endpoint == "dummy"
    assert app_settings.azure_openai is not None
    
    # Validate API payload structure
    payload = app_settings.datasource.construct_payload_configuration()
    assert payload["type"] == "elasticsearch"
    assert payload["parameters"] is not None
    assert payload["parameters"]["endpoint"] == "dummy"
    print(payload)


def test_dotenv_gpt_5_1_with_azure_search_success(app_settings):
    assert app_settings.azure_openai.is_reasoning_model is True
    assert app_settings.base_settings.datasource_type == "AzureCognitiveSearch"
    assert app_settings.datasource is not None
    assert app_settings.datasource.endpoint == "https://search_service.search.azure.us"
    assert app_settings.datasource.query_type == "vector_semantic_hybrid"
    assert app_settings.datasource.uses_vector_search is True
    assert app_settings.datasource.uses_semantic_search is True

    
    

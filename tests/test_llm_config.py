from pytest import MonkeyPatch
from src.llm.config import ModelConfiguration, ModelRole, load_model_configuration


def test_model_configuration_uses_default_models(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_CONTRACT_BUILDER", raising=False)
    monkeypatch.delenv("MODEL_PLANNER", raising=False)
    monkeypatch.delenv("MODEL_CONTEXT_GATHERER", raising=False)
    monkeypatch.delenv("MODEL_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("MODEL_REVIEWER", raising=False)
    monkeypatch.delenv("MODEL_SUMMARIZER", raising=False)

    model_configuration = load_model_configuration()

    assert model_configuration.model_for_role(ModelRole.CONTRACT_BUILDER) == "claude-opus-4-7"
    assert model_configuration.model_for_role(ModelRole.CONTEXT_GATHERER) == (
        "claude-haiku-4-5-20251001"
    )
    assert model_configuration.model_for_role(ModelRole.IMPLEMENTATION) == "claude-sonnet-4-6"


def test_model_configuration_uses_environment_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_REVIEWER", "review-model")

    model_configuration = load_model_configuration()

    assert model_configuration.model_for_role(ModelRole.REVIEWER) == "review-model"


def test_model_configuration_is_frozen() -> None:
    model_configuration = ModelConfiguration(
        contract_builder_model="a",
        planner_model="b",
        complexity_assessor_model="c",
        context_gatherer_model="d",
        implementation_model="e",
        reviewer_model="f",
        summarizer_model="g",
    )

    try:
        model_configuration.reviewer_model = "changed"
    except Exception:
        return

    raise AssertionError("ModelConfiguration should be immutable")

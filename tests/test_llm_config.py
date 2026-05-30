from pytest import MonkeyPatch
from src.config import ModelRole, load_settings


def test_settings_use_default_models(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv('MODEL_CONTRACT_BUILDER', raising=False)
    monkeypatch.delenv('MODEL_PLANNER', raising=False)
    monkeypatch.delenv('MODEL_CONTEXT_GATHERER', raising=False)
    monkeypatch.delenv('MODEL_IMPLEMENTATION', raising=False)
    monkeypatch.delenv('MODEL_REVIEWER', raising=False)
    monkeypatch.delenv('MODEL_SUMMARIZER', raising=False)

    settings = load_settings()

    assert settings.model_for_role(ModelRole.CONTRACT_BUILDER) == 'claude-opus-4-7'
    assert settings.model_for_role(ModelRole.CONTEXT_GATHERER) == 'claude-haiku-4-5-20251001'
    assert settings.model_for_role(ModelRole.IMPLEMENTATION) == 'claude-sonnet-4-6'
    assert settings.context_limit_for_role(ModelRole.IMPLEMENTATION) == 200_000


def test_settings_use_environment_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv('MODEL_REVIEWER', 'claude-haiku-4-5-20251001')

    settings = load_settings()

    assert settings.model_for_role(ModelRole.REVIEWER) == 'claude-haiku-4-5-20251001'


def test_settings_rejects_unknown_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv('MODEL_REVIEWER', 'no-such-model-xyz')

    try:
        load_settings()
    except ValueError as error:
        assert 'no-such-model-xyz' in str(error)
        return

    raise AssertionError('load_settings should reject unknown model ids')


def test_settings_model_entry_exposes_cost(monkeypatch: MonkeyPatch) -> None:
    settings = load_settings()

    opus_entry = settings.model_entry('claude-opus-4-7')

    assert opus_entry.input_price_usd_per_mtok == 15.0
    assert opus_entry.output_price_usd_per_mtok == 75.0


def test_settings_is_frozen() -> None:
    settings = load_settings()

    try:
        settings.workspace_root = 'changed'
    except Exception:
        return

    raise AssertionError('Settings should be immutable')

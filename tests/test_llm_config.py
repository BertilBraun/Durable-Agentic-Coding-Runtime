from pytest import MonkeyPatch
from src.config import ModelRole, load_settings


def _clear_model_environment(monkeypatch: MonkeyPatch) -> None:
    for role in ModelRole:
        monkeypatch.delenv(f'MODEL_{role.name}', raising=False)


def test_settings_use_default_models(monkeypatch: MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)

    settings = load_settings()

    assert settings.model_for_role(ModelRole.CONTRACT_BUILDER) == 'claude-opus-4-7'
    assert settings.model_for_role(ModelRole.CONTEXT_GATHERER) == 'claude-haiku-4-5-20251001'
    assert settings.model_for_role(ModelRole.IMPLEMENTATION) == 'claude-sonnet-4-6'
    assert settings.context_limit_for_role(ModelRole.IMPLEMENTATION) == 200_000


def test_settings_use_environment_override(monkeypatch: MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv('MODEL_REVIEWER', 'claude-haiku-4-5-20251001')

    settings = load_settings()

    assert settings.model_for_role(ModelRole.REVIEWER) == 'claude-haiku-4-5-20251001'


def test_settings_rejects_unknown_model_before_family_check(monkeypatch: MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv('MODEL_REVIEWER', 'no-such-model-xyz')
    monkeypatch.setenv('MODEL_PLANNER', 'gemini-3.1-flash-lite')

    try:
        load_settings()
    except ValueError as error:
        assert 'no-such-model-xyz' in str(error)
        return

    raise AssertionError('load_settings should reject unknown model ids')


def test_settings_rejects_mixed_model_families(monkeypatch: MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv('MODEL_PLANNER', 'gemini-3.1-flash-lite')

    try:
        load_settings()
    except ValueError as error:
        assert 'same model family' in str(error)
        assert 'claude' in str(error)
        assert 'gemini' in str(error)
        return

    raise AssertionError('load_settings should reject mixed model families')


def test_settings_model_entry_exposes_cost(monkeypatch: MonkeyPatch) -> None:
    settings = load_settings()

    opus_entry = settings.model_entry('claude-opus-4-7')

    assert opus_entry.input_price_usd_per_mtok == 15.0
    assert opus_entry.output_price_usd_per_mtok == 75.0


def test_planning_roles_get_a_reasoning_effort(monkeypatch: MonkeyPatch) -> None:
    for role in (
        ModelRole.CONTRACT_BUILDER,
        ModelRole.PLANNER,
    ):
        monkeypatch.delenv(f'REASONING_EFFORT_{role.name}', raising=False)

    settings = load_settings()

    assert settings.reasoning_effort_for_role(ModelRole.CONTRACT_BUILDER) is not None
    assert settings.reasoning_effort_for_role(ModelRole.PLANNER) is not None


def test_non_planning_roles_have_reasoning_off_by_default(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv('REASONING_EFFORT_REVIEWER', raising=False)

    settings = load_settings()

    assert settings.reasoning_effort_for_role(ModelRole.REVIEWER) is None


def test_settings_is_frozen() -> None:
    settings = load_settings()

    try:
        settings.artifacts_root = 'changed'
    except Exception:
        return

    raise AssertionError('Settings should be immutable')

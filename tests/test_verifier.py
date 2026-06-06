import pytest
from src.activities import verifier as verifier_module
from src.activities.workspace_manager import HostWorkspace, ToolExecutionRequest, ToolResult
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext


def _workspace() -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
    )


@pytest.mark.asyncio
async def test_run_anchor_tests_runs_repro_then_regression_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        commands.append(request.tool.tool_name.value)
        return ToolResult(stdout='ok', stderr='', exit_code=0, truncated=False)

    async def fake_assert_present(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='1', stderr='', exit_code=0, truncated=False)

    monkeypatch.setattr(verifier_module, 'run_tool', fake_run_tool)
    monkeypatch.setattr('src.activities.test_protection.run_tool', fake_assert_present)

    results = await verifier_module.run_anchor_tests(
        _workspace(),
        RepoIndex(),
        ReproductionContext(
            repro_target='pkg/tests/test_repro.py::test_round_trip',
            failure_evidence='boom',
            regression_test_files=['pkg/tests/test_existing.py'],
        ),
        restore_regression_files=True,
    )

    assert commands == ['run_shell', 'run_tests', 'run_tests']
    assert [result.passed for result in results] == [True, True]
    assert results[0].command == 'pkg/tests/test_repro.py::test_round_trip'
    assert results[1].command == 'pkg/tests/test_existing.py'


@pytest.mark.asyncio
async def test_run_anchor_tests_fails_repro_with_no_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='passed', stderr='', exit_code=0, truncated=False)

    async def fake_no_assertion(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='0', stderr='', exit_code=1, truncated=False)

    monkeypatch.setattr(verifier_module, 'run_tool', fake_run_tool)
    monkeypatch.setattr('src.activities.test_protection.run_tool', fake_no_assertion)

    results = await verifier_module.run_anchor_tests(
        _workspace(),
        RepoIndex(),
        ReproductionContext(repro_target='t.py::t', failure_evidence='x'),
        restore_regression_files=False,
    )

    assert results[0].passed is False
    assert 'no assertion' in results[0].stderr_summary


@pytest.mark.asyncio
async def test_run_anchor_tests_skips_restore_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        commands.append(request.tool.tool_name.value)
        return ToolResult(stdout='', stderr='', exit_code=1, truncated=False)

    monkeypatch.setattr(verifier_module, 'run_tool', fake_run_tool)

    results = await verifier_module.run_anchor_tests(
        _workspace(),
        RepoIndex(),
        ReproductionContext(repro_target='t.py::t', failure_evidence='x'),
        restore_regression_files=False,
    )

    assert commands == ['run_tests']
    assert results[0].passed is False

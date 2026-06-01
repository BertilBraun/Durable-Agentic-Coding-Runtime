from __future__ import annotations

import json
from typing import TypeVar, cast

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel
from src.llm.client import Message

StructuredFakeOutput = TypeVar('StructuredFakeOutput', bound=BaseModel)


class FakeOpenAICompletions:
    async def create(self, **keyword_arguments: object) -> ChatCompletion:
        return ChatCompletion.model_validate(
            {
                'id': 'chatcmpl-fake',
                'object': 'chat.completion',
                'created': 0,
                'model': 'fake-completion-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': '{}'},
                    }
                ],
                'usage': {
                    'prompt_tokens': 1,
                    'completion_tokens': 1,
                    'total_tokens': 2,
                },
            }
        )

    async def parse(
        self,
        response_format: type[StructuredFakeOutput],
        **keyword_arguments: object,
    ) -> ParsedChatCompletion[StructuredFakeOutput]:
        raw_messages = cast(list[dict[str, str]], keyword_arguments['messages'])
        messages = [Message.model_validate(raw_message) for raw_message in raw_messages]
        content = _fake_structured_content(response_format=response_format, messages=messages)
        parsed = response_format.model_validate_json(content)
        parsed_completion_type = ParsedChatCompletion[response_format]
        return parsed_completion_type.model_validate(
            {
                'id': 'chatcmpl-fake',
                'object': 'chat.completion',
                'created': 0,
                'model': 'fake-structured-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': content,
                            'parsed': parsed.model_dump(mode='json'),
                        },
                    }
                ],
                'usage': {
                    'prompt_tokens': _token_count(messages),
                    'completion_tokens': len(content.split()),
                    'total_tokens': _token_count(messages) + len(content.split()),
                },
            }
        )


class FakeOpenAIChat:
    def __init__(self, completions: FakeOpenAICompletions) -> None:
        self.completions = completions


class FakeOpenAIBeta:
    def __init__(self, completions: FakeOpenAICompletions) -> None:
        self.chat = FakeOpenAIChat(completions)


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.completions = FakeOpenAICompletions()
        self.chat = FakeOpenAIChat(self.completions)
        self.beta = FakeOpenAIBeta(self.completions)


def _fake_structured_content(response_format: type[BaseModel], messages: list[Message]) -> str:
    match response_format.__name__:
        case 'TaskContract':
            return (
                '{"task_type":"bugfix","goal":"Run the smoke workflow",'
                '"acceptance_criteria":["Workflow completes"],"non_goals":[],'
                '"affected_areas":["smoke"],"risk_areas":[],"tests_expected":[],'
                '"open_questions":[]}'
            )
        case 'Plan':
            return (
                '{"summary":"Smoke test plan","steps":[{"id":"step_1",'
                '"goal":"Add a deterministic smoke patch","target_files":["app.py","test_app.py"],'
                '"tests_to_run":["pytest -q"],'
                '"expected_result":"Patch changes code and tests pass",'
                '"risk":"low","requires_human_approval":false}],'
                '"integration_tests":["pytest -q"],'
                '"definition_of_done":["Diff is non-empty","Smoke test passes"]}'
            )
        case 'ImplementationAgentTurn':
            if _implementation_has_test_observation(messages):
                return (
                    '{"done":true,"worker_result":{"status":"success","patch_id":"smoke",'
                    '"diff_summary":"Added smoke subtract function and test.",'
                    '"tests_run":[],"test_results":[],"discovered_issues":[],'
                    '"confidence":"high","replan_suggestion":null},"tool_calls":[]}'
                )
            return _implementation_tool_turn()
        case _:
            raise ValueError(f'No fake structured response for {response_format.__name__}')


def _implementation_has_test_observation(messages: list[Message]) -> bool:
    return any('tool=run_tests' in message.content for message in messages)


def _implementation_tool_turn() -> str:
    patch = (
        'diff --git a/app.py b/app.py\n'
        '--- a/app.py\n'
        '+++ b/app.py\n'
        '@@ -1,2 +1,5 @@\n'
        ' def add(first_number: int, second_number: int) -> int:\n'
        '     return first_number + second_number\n'
        '+\n'
        '+def subtract(first_number: int, second_number: int) -> int:\n'
        '+    return first_number - second_number\n'
        'diff --git a/test_app.py b/test_app.py\n'
        'new file mode 100644\n'
        '--- /dev/null\n'
        '+++ b/test_app.py\n'
        '@@ -0,0 +1,5 @@\n'
        '+from app import subtract\n'
        '+\n'
        '+\n'
        '+def test_subtract() -> None:\n'
        '+    assert subtract(3, 1) == 2\n'
    )
    escaped_patch = json.dumps(patch)
    return (
        '{"done":false,"worker_result":null,"tool_calls":['
        f'{{"tool_name":"apply_patch","patch":{escaped_patch}}},'
        '{"tool_name":"run_tests","command":"pytest -q",'
        '"timeout_seconds":60,"directory":"."}'
        ']}'
    )


def _token_count(messages: list[Message]) -> int:
    return sum(len(message.content.split()) for message in messages)

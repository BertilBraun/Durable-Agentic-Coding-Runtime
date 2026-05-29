from __future__ import annotations

from src.activities.temporal import durable_activity
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.approval import ComplexityVerdict
from src.models.task import TaskContract

# TODO overarching - can we abstract a llm call / llm structured generation into an activity itself. We use that so often and it should always be an activity, so we get retries, timeouts, etc. for free and we don't have to repeat the same code all over the place. We can just pass in the messages, the role and the output type and get back the structured output with all the activity goodness.
# Also extract into that activity the concept of role (and the system prompt addition - then again, why dont we define the system prompt where it is used?? I.e. the complete prompt for the complecity verifier should be here in this file, not split up in all kinds of locations - or alternatively, we should have ONE central location for all prompts, but then all all prompts should be there - probably for the future for A/B testing, for now I think defining the prompts where they are used is the way to go) - that is always necessary but we repeat that code all over the place / the llm client itself should not be aware of the roles - i.e. remove the role from the llm client as well.
# See reviewer, planner, implementation etc. for examples of this code being repeated all over the place


@durable_activity(retries=2, timeout=120, backoff_seconds=10)
async def assess_complexity(contract: TaskContract) -> ComplexityVerdict:
    llm_client = LLMClient()
    return await llm_client.generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[
            Message(
                role='system',
                content=system_prompt_for_role(ModelRole.COMPLEXITY_ASSESSOR),
            ),
            Message(role='user', content=contract.model_dump_json()),
        ],
        output_type=ComplexityVerdict,
    )

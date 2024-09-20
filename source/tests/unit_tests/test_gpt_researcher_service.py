import asyncio
import pytest
from source.code.services.object_model.agents.GPTResearcherAgent import get_research_response
# from services.object_model.agents.GPTResearcherAgent import get_research_response


@pytest.mark.asyncio
class TestGPTResearcherServices:

    async def async_setup(self):
        await asyncio.sleep(1)
        self.resource = "initialized"

    async def async_teardown(self):
        await asyncio.sleep(1)
        self.resource = None

    def test_research_response(self):
        research_question_file_path = "../tests/data/inputs/research_question.prompt"
        output_file_path = "../tests/data/outputs/test_report1.txt"

        asyncio.run(get_research_response(research_question_file_path, output_file_path))


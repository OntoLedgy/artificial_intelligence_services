import asyncio
import pytest

from services.object_model.agents.GPTResearcherAgent import GPTResearcherAgent

@pytest.mark.asyncio
class TestGPTResearcherServices:

    async def async_setup(self):
        await asyncio.sleep(1)
        self.resource = "initialized"

    async def async_teardown(self):
        await asyncio.sleep(1)
        self.resource = None

    async def test_research_response(
            self):

        research_question_file = open("../data/inputs/research_question.prompt")

        research_question = research_question_file.read()

        researcher_service = GPTResearcherAgent(
            research_question
            )

        response = await researcher_service.generate_report()

        file = open("../data/outputs/test_report1.txt", "w")
        file.write(response)
        file.close()



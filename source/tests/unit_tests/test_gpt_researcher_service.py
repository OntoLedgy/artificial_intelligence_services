import asyncio
import os
import pytest

from services.agents.orchestrators.gpt_researcher_orchestrator import orchestrate_gpt_research


@pytest.mark.asyncio
class TestGPTResearcherServices:
    async def async_setup(self):
        await asyncio.sleep(1)
        self.resource = "initialized"

    async def async_teardown(self):
        await asyncio.sleep(1)
        self.resource = None

    def test_research_response(self,
                               inputs_folder_absolute_path,
                               outputs_folder_absolute_path):
        
        research_question_file_path = os.path.join(
                inputs_folder_absolute_path,
                "research_question.prompt"
                )
        
        output_file_path = os.path.join(
                outputs_folder_absolute_path,
                "test_report3.md"
                )

        asyncio.run(
            orchestrate_gpt_research(research_question_file_path, output_file_path)
        )

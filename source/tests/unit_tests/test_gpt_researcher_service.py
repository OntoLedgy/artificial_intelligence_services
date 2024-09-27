import asyncio
import pytest

from services.orchestrators.orchestrate_gpt_researcher import orchestrate_gpt_research


@pytest.mark.asyncio
class TestGPTResearcherServices:
    
    async def async_setup(
            self):
        await asyncio.sleep(
            1)
        self.resource = "initialized"
    
    
    async def async_teardown(
            self):
        await asyncio.sleep(
            1)
        self.resource = None
    
    
    def test_research_response(
            self):
        research_question_file_path = "../tests/data/inputs/research_question.prompt"
        output_file_path = "../tests/data/outputs/test_report3.md"
        
        asyncio.run(
                orchestrate_gpt_research(
                    research_question_file_path,
                    output_file_path)
                )

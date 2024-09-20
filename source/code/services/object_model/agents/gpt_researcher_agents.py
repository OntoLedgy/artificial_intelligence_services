from gpt_researcher import GPTResearcher
import asyncio


async def get_research_response(research_question_file, output_file):
    with open(research_question_file, 'r') as f:
        research_question = f.read()

    researcher_service = GPTResearcherAgents(research_question)
    response = await researcher_service.generate_report()

    with open(output_file, 'w') as f:
        f.write(response)

class GPTResearcherAgents:
    def __init__(

            self,
            query,
            report_type="research_report"
    ):
        self.researcher = None
        self.query = query
        self.report_type = report_type

    async def generate_report(

            self
    ):
        self.researcher = GPTResearcher(
            agent="gpt-4o",
            query=self.query,
            report_type=self.report_type)

        report = await self._get_report()

        return report

    async def _get_report(
            self
    ) -> str:
        research_result = await self.researcher.conduct_research()
        report = await self.researcher.write_report()

        return report

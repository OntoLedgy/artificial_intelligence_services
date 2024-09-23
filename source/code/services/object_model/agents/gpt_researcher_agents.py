from gpt_researcher import GPTResearcher
import asyncio





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

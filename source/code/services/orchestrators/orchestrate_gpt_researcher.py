from services.object_model.agents.gpt_researcher_agents import GPTResearcherAgents


async def orchestrate_gpt_research(
        research_question_file,
        output_file):
    with open(research_question_file, 'r') as f:
        research_question = f.read()

    researcher_service = GPTResearcherAgents(research_question)
    response = await researcher_service.generate_report()

    with open(output_file, 'w') as f:
        f.write(response)
from source.code.services.object_model.agents.GPTResearcherAgent import get_research_response
from source.code.services.object_model.clients.OpenAiClient import OpenAiClient

if __name__ == '__main__':
    prompt = "Explain the theory of relativity in simple terms."

    openai_service = \
        OpenAiClient(
            api_key='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')

    response = openai_service.get_response(prompt)

    print(response)


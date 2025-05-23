

class TestOpenAiServices:
    def test_gpt_response(self,
                          openai_client):
        prompt = "Explain the theory of relativity in simple terms."

        response = openai_client.get_response(prompt)

        print(response)

        assert response is not None

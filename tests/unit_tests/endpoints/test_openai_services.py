#TODO: needs to be fixed to address error
"""OpenAIConfigurations Configuration Validated and Loaded: api_key='sk-proj-3jvM4IHUxnX7ZoPSkirzT3BlbkFJptBI6eMAq7yxhjNs1ln8' openai_organisation='org-3jvM4IHUxnX7ZoPSkirzT3BlbkFJptBI6eMAq7yxhjNs1ln8' openai_project='proj-3jvM4IHUxnX7ZoPSkirzT3BlbkFJptBI6eMAq7yxhjNs1ln8' openai_model='gpt-4o' max_tokens=100 temperature=0.5 top_p=1
FAILED [100%]Error during API call:

You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface.

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`

A detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742

None
"""


class TestOpenAiServices:
    def test_gpt_response(self,
                          openai_client):
        prompt = "Explain the theory of relativity in simple terms."

        response = openai_client.get_response(prompt)

        print(response)

        assert response is not None

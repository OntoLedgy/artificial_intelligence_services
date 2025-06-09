
from agents.coding.prompt_templates.coding_prompt_template import SYSTEM_PROMPT
from agents.coding.tools.python_repl_tool import python_repl
from agents.coding.tools.python_repl_tool import python_tool
from agents.coding.tools.results_validator import validation_tool
from agents.coding.tools.results_validator import validator
from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations

from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

from typing import Dict, \
    Any

from langchain.chat_models import init_chat_model

class AdvancedCodeAgent:
    def __init__(self):

        # Initialize ChatOpenAI (reads OPENAI_API_KEY from env)
        self.llm = init_chat_model(
            model=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI,
            temperature=0,
            max_tokens=4000
        )
        # Use langgraph pre-built React agent with system prompt
        self.agent = create_react_agent(
            model=self.llm,
            tools=[python_tool, validation_tool],
            prompt=SYSTEM_PROMPT,
            version="v2"
        )

        self.python_repl = python_repl
        self.validator = validator

    def run(self, query: str) -> str:
        """Run the coding agent on the given query and return the final answer."""
        try:
            # Initialize with a human message
            input_messages = [HumanMessage(content=query)]
            result = self.agent.invoke({"messages": input_messages})
            # Extract messages list from result
            messages = result.get("messages", [])
            if messages:
                # Return content of last message (final answer)
                return getattr(messages[-1], "content", str(messages[-1]))
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    def validate_last_result(self, description: str, validation_params: Dict[str, Any]) -> str:
        """Manually validate the last computation result"""
        if 'test_cases' in validation_params:
            return self.validator.validate_algorithm_correctness(description, validation_params['test_cases'])
        elif 'expected_structure' in validation_params:
            return self.validator.validate_data_analysis(description, validation_params['expected_structure'])
        else:
            return self.validator.validate_mathematical_result(description, validation_params)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions"""
        history = self.python_repl.get_execution_history()
        return {
            'total_executions': len(history),
            'successful_executions': len([h for h in history if not h['error']]),
            'failed_executions': len([h for h in history if h['error']]),
            'execution_details': history
        }
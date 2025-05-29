import os

from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations

from langchain.agents import create_react_agent, AgentExecutor

from typing import Dict, \
    Any

from langchain_openai import ChatOpenAI

from agents.coding.prompt_templates.claude_coding import prompt
from agents.coding.tools.python_repl_tool import PythonREPLTool
from agents.coding.tools.results_validator import ResultValidator
from langchain.tools import Tool

python_repl = PythonREPLTool()
validator = ResultValidator(python_repl)



python_tool = Tool(
    name="python_repl",
    description="Execute Python code and return both the code and its output. Maintains state between executions.",
    func=python_repl.run
)

validation_tool = Tool(
    name="result_validator",
    description="Validate the results of previous computations with specific test cases and expected properties.",
    func=lambda query: validator.validate_mathematical_result(query, {})
)


class AdvancedCodeAgent:
    def __init__(self, llm_api_key=None):
        if llm_api_key:
            os.environ["ANTHROPIC_API_KEY"] = llm_api_key

        self.llm = ChatOpenAI(
            model=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT_4O_MINI,
            temperature=0,
            max_tokens=4000
        )
        #TODO: replace with langgraph create react agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=[python_tool, validation_tool],
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[python_tool, validation_tool],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            return_intermediate_steps=True
        )

        self.python_repl = python_repl
        self.validator = validator

    def run(self, query: str) -> str:
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
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
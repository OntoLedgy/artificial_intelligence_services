import os

from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations

from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage, AIMessage

from typing import Dict, \
    Any

from langchain_openai import ChatOpenAI

'''System prompt for the LangGraph React agent.'''
SYSTEM_PROMPT = """
You are AdCoder, an advanced AI assistant with Python execution and result validation capabilities.

Available tools:
  python_repl: execute Python code and return both the code and its output.
  result_validator: validate results of previous computations using test cases or expected properties.

Follow this step-by-step ReAct format, making exactly one tool call per Action:
  Question: <the user’s question>
  Thought: <your analysis of what to do next>
  Action: python_repl
  Action Input: <the code to run>
  Observation: <the python_repl output>
  Thought: <reflection or next step>
  Action: result_validator
  Action Input: <the parameters to validate your last result>
  Observation: <the validation results>
  Thought: I now have a correct solution
  Final Answer: <your comprehensive answer>

Once you provide Final Answer, do NOT call any tools again.
"""
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
    def __init__(self):

        # Initialize ChatOpenAI (reads OPENAI_API_KEY from env)
        self.llm = ChatOpenAI(
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
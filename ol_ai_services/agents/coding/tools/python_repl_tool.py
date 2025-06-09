import sys
import io
import re
import json
import textwrap
from typing import Dict, Any, List
from langchain.tools import Tool

class PythonREPLTool:
    def __init__(self):
        self.globals_dict = {
            '__builtins__': __builtins__,
            'json': json,
            're': re
        }
        self.locals_dict = {}
        self.execution_history = []

    def run(self, code: str) -> str:
        # Remove any Markdown code fences and standalone python tags anywhere in the code
        lines = code.splitlines()
        filtered = []
        for line in lines:
            s = line.strip().lower()
            # Skip fence markers and lone language specifier
            if s.startswith("```") or s == "python":
                continue
            filtered.append(line)
        code = "\n".join(filtered)
        # Remove common leading indentation for proper execution
        code = textwrap.dedent(code)
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = captured_output = io.StringIO()
            sys.stderr = captured_error = io.StringIO()

            execution_result = None

            try:
                result = eval(code, self.globals_dict, self.locals_dict)
                execution_result = result
                if result is not None:
                    print(result)
            except SyntaxError:
                exec(code, self.globals_dict, self.locals_dict)

            output = captured_output.getvalue()
            error_output = captured_error.getvalue()

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            self.execution_history.append({
                'code': code,
                'output': output,
                'result': execution_result,
                'error': error_output
            })

            response = f"**Code Executed:**\n```python\n{code}\n```\n\n"
            if error_output:
                response += f"**Errors/Warnings:**\n{error_output}\n\n"
            response += f"**Output:**\n{output if output.strip() else 'No console output'}"

            if execution_result is not None and not output.strip():
                response += f"\n**Return Value:** {execution_result}"

            return response

        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            error_info = f"**Code Executed:**\n```python\n{code}\n```\n\n**Runtime Error:**\n{str(e)}\n**Error Type:** {type(e).__name__}"

            self.execution_history.append({
                'code': code,
                'output': '',
                'result': None,
                'error': str(e)
            })

            return error_info

    def get_execution_history(self) -> List[Dict[str, Any]]:
        return self.execution_history

    def clear_history(self):
        self.execution_history = []


python_repl = PythonREPLTool()


python_tool = Tool(
    name="python_repl",
    description="Execute Python code and return both the code and its output. Maintains state between executions.",
    func=python_repl.run
)

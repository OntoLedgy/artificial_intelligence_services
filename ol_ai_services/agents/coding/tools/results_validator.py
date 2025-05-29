from typing import Dict, Any, List

from ol_ai_services.agents.coding.tools.python_repl_tool import PythonREPLTool

class ResultValidator:
    def __init__(self, python_repl: PythonREPLTool):
        self.python_repl = python_repl

    def validate_mathematical_result(self, description: str, expected_properties: Dict[str, Any]) -> str:
        """Validate mathematical computations"""
        # Sanitize description: remove fences and initial python tag lines
        desc_lines = description.splitlines()
        if desc_lines and desc_lines[0].strip().startswith("```"):
            desc_lines = desc_lines[1:]
        if desc_lines and desc_lines[-1].strip().startswith("```"):
            desc_lines = desc_lines[:-1]
        if desc_lines and desc_lines[0].strip().lower().startswith("python"):
            desc_lines = desc_lines[1:]
        # Inline description for comment
        desc_inline = " ".join(line.strip() for line in desc_lines) if desc_lines else ""
        validation_code = f"""
            # Validation for: {desc_inline}
            validation_results = {{}}

            # Get the last execution results
            history = {self.python_repl.execution_history}
            if history:
                last_execution = history[-1]
                print(f"Last execution output: {{last_execution['output']}}")

                # Extract numbers from the output
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', last_execution['output'])
                if numbers:
                    numbers = [float(n) for n in numbers]
                    validation_results['extracted_numbers'] = numbers

                    # Validate expected properties
                    for prop, expected_value in {expected_properties}.items():
                        if prop == 'count':
                            actual_count = len(numbers)
                            validation_results[f'count_check'] = actual_count == expected_value
                            print(f"Count validation: Expected {{expected_value}}, Got {{actual_count}}")
                        elif prop == 'max_value':
                            if numbers:
                                max_val = max(numbers)
                                validation_results[f'max_check'] = max_val <= expected_value
                                print(f"Max value validation: {{max_val}} <= {{expected_value}} = {{max_val <= expected_value}}")
                        elif prop == 'min_value':
                            if numbers:
                                min_val = min(numbers)
                                validation_results[f'min_check'] = min_val >= expected_value
                                print(f"Min value validation: {{min_val}} >= {{expected_value}} = {{min_val >= expected_value}}")
                        elif prop == 'sum_range':
                            if numbers:
                                total = sum(numbers)
                                min_sum, max_sum = expected_value
                                validation_results[f'sum_check'] = min_sum <= total <= max_sum
                                print(f"Sum validation: {{min_sum}} <= {{total}} <= {{max_sum}} = {{min_sum <= total <= max_sum}}")

            # Separator and validation summary header
            print()
            print("Validation Summary:")
            for key, value in validation_results.items():
                print(f"{{key}}: {{value}}")

            validation_results
        """
        return self.python_repl.run(validation_code)

    def validate_data_analysis(self, description: str, expected_structure: Dict[str, Any]) -> str:
        """Validate data analysis results"""
        validation_code = f"""
# Data Analysis Validation for: {description}
validation_results = {{}}

# Check if required variables exist in global scope
required_vars = {list(expected_structure.keys())}
existing_vars = []

for var_name in required_vars:
    if var_name in globals():
        existing_vars.append(var_name)
        var_value = globals()[var_name]
        validation_results[f'{{var_name}}_exists'] = True
        validation_results[f'{{var_name}}_type'] = type(var_value).__name__

        # Type-specific validations
        if isinstance(var_value, (list, tuple)):
            validation_results[f'{{var_name}}_length'] = len(var_value)
        elif isinstance(var_value, dict):
            validation_results[f'{{var_name}}_keys'] = list(var_value.keys())
        elif isinstance(var_value, (int, float)):
            validation_results[f'{{var_name}}_value'] = var_value

        print(f"✓ Variable '{{var_name}}' found: {{type(var_value).__name__}} = {{var_value}}")
    else:
        validation_results[f'{{var_name}}_exists'] = False
        print(f"✗ Variable '{{var_name}}' not found")

print(f"\nFound {{len(existing_vars)}}/{{len(required_vars)}} required variables")

# Additional structure validation
for var_name, expected_type in {expected_structure}.items():
    if var_name in globals():
        actual_type = type(globals()[var_name]).__name__
        validation_results[f'{{var_name}}_type_match'] = actual_type == expected_type
        print(f"Type check '{{var_name}}': Expected {{expected_type}}, Got {{actual_type}}")

validation_results
"""
        return self.python_repl.run(validation_code)

    def validate_algorithm_correctness(self, description: str, test_cases: List[Dict[str, Any]]) -> str:
        """Validate algorithm implementations with test cases"""
        validation_code = f"""
# Algorithm Validation for: {description}
validation_results = {{}}
test_results = []

test_cases = {test_cases}

for i, test_case in enumerate(test_cases):
    test_name = test_case.get('name', f'Test {{i+1}}')
    input_val = test_case.get('input')
    expected = test_case.get('expected')
    function_name = test_case.get('function')

    print(f"\nRunning {{test_name}}:")
    print(f"Input: {{input_val}}")
    print(f"Expected: {{expected}}")

    try:
        if function_name and function_name in globals():
            func = globals()[function_name]
            if callable(func):
                if isinstance(input_val, (list, tuple)):
                    result = func(*input_val)
                else:
                    result = func(input_val)

                passed = result == expected
                test_results.append({{
                    'test_name': test_name,
                    'input': input_val,
                    'expected': expected,
                    'actual': result,
                    'passed': passed
                }})

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"Actual: {{result}}")
                print(f"Status: {{status}}")
            else:
                print(f"✗ ERROR: '{{function_name}}' is not callable")
        else:
            print(f"✗ ERROR: Function '{{function_name}}' not found")

    except Exception as e:
        print(f"✗ ERROR: {{str(e)}}")
        test_results.append({{
            'test_name': test_name,
            'error': str(e),
            'passed': False
        }})

# Summary
passed_tests = sum(1 for test in test_results if test.get('passed', False))
total_tests = len(test_results)
validation_results['tests_passed'] = passed_tests
validation_results['total_tests'] = total_tests
validation_results['success_rate'] = passed_tests / total_tests if total_tests > 0 else 0

print(f"\n=== VALIDATION SUMMARY ===")
print(f"Tests passed: {{passed_tests}}/{{total_tests}}")
print(f"Success rate: {{validation_results['success_rate']:.1%}}")

test_results
"""
        return self.python_repl.run(validation_code)
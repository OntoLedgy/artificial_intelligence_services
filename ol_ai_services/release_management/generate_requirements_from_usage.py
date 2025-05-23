import ast
import os
import sys

def get_imports(path):
    
    imports = set()
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=file_path)
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
    return imports

def is_stdlib_module(module_name):
    
    try:
        # Python 3.10+: use stdlib_module_names
        import sys
        if hasattr(sys, "stdlib_module_names"):
            return module_name in sys.stdlib_module_names
    except ImportError:
        pass
    # Fallback: attempt to locate module spec
    try:
        import importlib.util
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return "site-packages" not in spec.origin
    except Exception:
        pass
    return False

def main(root_path):
    imported = get_imports(root_path)
    third_party = sorted(m for m in imported if not is_stdlib_module(m))
    for pkg in third_party:
        print(pkg)

if __name__ == "__main__":
    project_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(project_dir)
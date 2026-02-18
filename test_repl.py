from langchain_experimental.utilities import PythonREPL

try:
    repl = PythonREPL()
    result = repl.run("print(10 * 10)")
    print(f"Result: {result.strip()}")
except Exception as e:
    print(f"Error: {e}")

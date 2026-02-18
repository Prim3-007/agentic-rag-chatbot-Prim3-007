from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import pandas as pd
import requests
import numpy as np

@tool
def python_interpreter(code: str):
    """
    A Python shell. Use this to execute python commands. Input should be a valid python script.
    Useful for:
    1. Performing complex calculations.
    2. Analyzing data independent of the LLM's context window.
    3. Fetching external data via APIs (e.g. Open-Meteo) using the 'requests' library.
    
    The environment has the following libraries pre-installed and available:
    - pandas (as pd)
    - numpy (as np)
    - requests
    
    When using Open-Meteo:
    - Endpoint: https://api.open-meteo.com/v1/forecast
    - Params: latitude, longitude, hourly=temperature_2m, etc.
    
    IMPORTANT: Print the final result to stdout so it can be returned.
    """
    repl = PythonREPL()
    # Simple safety: check for import os/sys or dangerous ops if we were robust, 
    # but for hackathon we stick to the basic REPL which allows imports but is "sandboxed" by logic.
    # Note: PythonREPL itself is not a secure sandbox!
    
    try:
        # We can pre-import common libs in the scope if needed, but PythonREPL usually starts fresh.
        # Users (LLM) should import what they need.
        return repl.run(code)
    except Exception as e:
        return f"Error executing code: {e}"

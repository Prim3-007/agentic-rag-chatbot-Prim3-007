from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.sandbox import python_interpreter

def is_safe_code(code: str) -> bool:
    """
    Check if the code contains any unsafe imports or functions.
    Returns True if code is safe, False otherwise.
    """
    unsafe_patterns = [
        "import os", "import sys", "import subprocess", "import shutil", "import pty",
        "__import__",
        "eval(", "exec(",
        "open("
    ]
    for pattern in unsafe_patterns:
        if pattern in code:
            return False
    return True

SECURITY_BLOCK_MSG = "SECURITY ALERT: Execution blocked due to unsafe code pattern."

@tool
def analyze_weather(location: str) -> str:
    """
    Analyze the weather for a specific location using historical data or forecast.
    Retrieves data from Open-Meteo and calculates insights like rolling averages or volatility.
    
    Args:
        location: The name of the city or location (e.g. "Topeka, KS" or "Berlin").
    """
    # 1. Geocoding (Simplified for hackathon: Ask LLM to pick a lat/long or use a fixed one if complex? 
    # Actually, Open-Meteo docs say we need lat/long. 
    # The 'python_interpreter' can use 'requests' to call the geocoding API too!)
    
    # We will use a sub-chain to generate the python code.
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Python developer. Write a script to analyze weather for {location}."),
        ("human", """
step 1: Use `requests` to call specific Open-Meteo Geocoding API to get lat/long for {location}.
step 2: Use `requests` to fetch daily temperature_2m_max data for the past 30 days from `https://archive-api.open-meteo.com/v1/archive`.
step 3: Load into pandas DataFrame.
step 4: Calculate 7-day rolling average and volatility (variance).
step 5: Print the results clearly.

IMPORTANT: 
- Output ONLY valid Python code. 
- Do not use markdown backticks.
- ALWAYS wrap the network calls in a try-except block to handle connection errors.
- Ensure `import requests` is at the top.
- If the API fails, print "Error: <details>" so the user knows.
""")
    ])
    
    chain = prompt | model | StrOutputParser()
    
    try:
        code = chain.invoke({"location": location})
        # Clean block formatting if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Clean block formatting if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Security Check
        if not is_safe_code(code):
            return SECURITY_BLOCK_MSG
            
        # Execute in sandbox
        result = python_interpreter(code)
        return f"Analysis Result:\n{result}"
    except Exception as e:
        return f"Failed to analyze weather: {e}"

from src.tools.weather import is_safe_code, analyze_weather

def test_security():
    print("Testing Security Filter...")
    
    # Safe code
    safe_code = "import pandas as pd\nprint(1+1)"
    assert is_safe_code(safe_code) == True
    print("✓ Safe code passed")
    
    # Unsafe code
    unsafe_codes = [
        "import os",
        "import sys",
        "exec('print(1)')",
        "eval('1+1')",
        "open('test.txt', 'w')",
        "__import__('os')"
    ]
    
    for code in unsafe_codes:
        if is_safe_code(code):
            print(f"FAILED: Unsafe code passed: {code}")
        else:
            print(f"✓ Unsafe code blocked: {code}")

if __name__ == "__main__":
    test_security()

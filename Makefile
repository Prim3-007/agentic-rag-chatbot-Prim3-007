.PHONY: sanity run clean install venv

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: venv

sanity:
	export OPENAI_API_KEY=sk-proj-dummy-key-for-import-validation; .venv/bin/python scripts/sanity_check.py

run:
	.venv/bin/uvicorn server:app --reload --port 8000

clean:
	rm -rf data temp_uploads artifacts __pycache__ .venv USER_MEMORY.md COMPANY_MEMORY.md sample_docs/test_company.txt
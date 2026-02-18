# CDF - Company Handbook 2026

## 1. Mission & Values
Community Dreams Foundation (CDF) is dedicated to democratizing AI for the open-source community.
**Motto**: "Code with Purpose, Deploy with Pride."
**Founded**: April 2024 by the "Prim3" team.

## 2. Engineering Standards
### Tech Stack
- **Backend**: Python 3.12+, FastAPI, LangChain.
- **Frontend**: Vanilla JS (No React/Vue bloat allowed).
- **Database**: PostgreSQL (Vector) and Redis (Cache).

### Code Style
- All Python code must be typed (mypy strict).
- Variable names must be `snake_case`.
- **Critical**: Never use `print()` in production; use `logger.info()`.

## 3. HR Policies
### Work Hours
- Core hours are 10:00 AM to 3:00 PM EST.
- Wednesday is a "No Meeting Day".

### Benefits
- **AI Allowance**: $200/month for LLM subscriptions (ChatGPT, Claude, Gemini).
- **Hardware**: M3 Max MacBook Pro for all Senior Engineers.
- **Vacation**: Unlimited (min 3 weeks mandatory).

## 4. Security Protocols
- **API Keys**: Must be stored in `.env` files, NEVER committed to Git.
- **Access Control**: Role-Based Access Control (RBAC) is enforced on all internal tools.
- **Incident Response**: Contact `security@cdf.internal` for any data breaches.

# Prospect Agent

A Streamlit app (Python 3.14) that finds sales prospects for **Baseten**, an AI inference platform. It looks for AI-native startups that recently raised money.

## Pipeline

Each stage feeds the next:

1. **Find**: pull recent funding rounds from SEC Form D filings.
2. **Filter**: keep only companies that match the ICP (ideal customer profile) settings.
3. **Qualify**: Claude reads the company's website and decides whether it is AI-native. It must include an evidence quote taken word-for-word from the site.
4. **Research**: write an account brief on the company.
5. **Draft**: write a cold email.

## Working with me

- I'm a beginner. Explain code in plain English: what it does and why, not just what changed.
- Build one step at a time. Finish and check one piece before starting the next, and don't build ahead of what I asked for.

## Rules

- **Secrets**: never put API keys, email addresses, or other credentials in code. Load them from `.env` with `python-dotenv`. `.env` is gitignored and must stay that way.
- **No invented facts**: every claim in an AI output (qualification, brief, email) must come from source data (Form D filing, company website, etc.) or be clearly labeled as an assumption. Prompts should tell Claude this explicitly.

## Environment

- The virtual environment is `.venv`. Activate it with `source .venv/bin/activate`, or call `.venv/bin/python` / `.venv/bin/pip` directly.
- Run the app with `streamlit run app.py`.

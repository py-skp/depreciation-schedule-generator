# Depreciation Schedule Generator

A Streamlit app to generate accurate SLM/WDV depreciation schedules with:
- exact-days or full-month conventions
- per-period salvage-floor capping
- final balancing to eliminate rounding drift
- PDF export

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
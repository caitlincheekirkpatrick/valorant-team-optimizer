# VALORANT Team Optimizer (Phase 1: Baseline)

**Goal:** Pre-match win probability per map using only pre-match information (rosters/agents, map, simple form priors).  
This repo ships a small curated dataset (no scraping). Future phases add SHAP, MLflow, and a Streamlit app with agent-comp optimization.

## Quickstart
```bash
make install
make features
make train

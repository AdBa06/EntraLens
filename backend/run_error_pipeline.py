#!/usr/bin/env python

import json
import pandas as pd
from error_analysis_utils import analyze_errors

# 1. Load your prompts directly from the Excel's PromptContent column
errors_path = "data/errors.xlsx"
df = pd.read_excel(errors_path)

if "PromptContent" not in df.columns or "SkillOutput" not in df.columns:
    raise ValueError("errors.xlsx must contain 'PromptContent' and 'SkillOutput' columns.")

# Dedupe so each prompt appears only once
questions = df["PromptContent"].dropna().astype(str).unique().tolist()

# 2. Run just the error pipeline
results = analyze_errors(questions, errors_path)

# 3. Dump out to a file
with open("data/error_results.json", "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2, ensure_ascii=False)

print(f"✅ Error pipeline complete — processed {len(results)} workloads")

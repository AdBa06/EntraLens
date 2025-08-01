# import os
# import pandas as pd
# from typing import List, Dict
# from classification_utils import classify_workloads

# def analyze_errors(
#     questions: List[str],
#     errors_path: str
# ) -> List[Dict]:
#     """
#     Mode 3: For each incoming question:
#       1) classify it into a workload (reusing classify_workloads)
#       2) look up the common errors for that question from errors.xlsx

#     Returns a list of dicts:
#       { "question": str, "workload": str, "errors": [str, ...], "percent": float }
#     """

#     # 1) Classify each question into workloads
#     wl_groups = classify_workloads(questions)

#     # 2) Invert to build a map: question -> workload
#     q_to_wl: Dict[str, str] = {}
#     for entry in wl_groups:
#         wl = entry["workload"]
#         for q in entry["questions"]:
#             q_to_wl[q] = wl

#     # 3) Load question-level error mappings from errors.xlsx
#     df = pd.read_excel(errors_path)
#     if "question" not in df.columns or "error" not in df.columns:
#         raise ValueError("errors.xlsx must contain 'question' and 'error' columns.")

#     errors_by_question = df.groupby("question")["error"].apply(list).to_dict()
#     total_errors = len(df)

#     # 4) Assemble per-question output
#     output: List[Dict] = []
#     for q in questions:
#         wl = q_to_wl.get(q)
#         errs = errors_by_question.get(q, [])
#         pct = round(100 * len(errs) / total_errors, 1) if total_errors else 0.0
#         output.append({
#             "question": q,
#             "workload": wl,
#             "errors": errs,
#             "percent": pct
#         })

#     return output

# error_analysis_utils.py



import os
import pandas as pd
from typing import List, Dict
from classification_utils import classify_workloads

def analyze_errors(
    questions: List[str],
    errors_path: str
) -> List[Dict]:
    """
    Mode 3: 
      1) classify each question into a workload (using classify_workloads)
      2) read errors.xlsx which has:
           - PromptContent (the question text)
           - SkillOutput   (the error message)
      3) for each row, map that exact PromptContent → SkillOutput
      4) for each workload, emit items = [{ question, errors: [SkillOutput] }, …]
    """

    # 1) Classify into workloads
    wl_groups = classify_workloads(questions)

    # 2) Load your Excel sheet
    df = pd.read_excel(errors_path)
    # Validate columns
    if "PromptContent" not in df.columns or "SkillOutput" not in df.columns:
        raise ValueError("errors.xlsx must contain 'PromptContent' and 'SkillOutput' columns.")

    # 3) Build direct map: PromptContent -> SkillOutput
    #    (if the same PromptContent appears multiple times, last one wins; 
    #     wrap into list for frontend consistency)
    records = df[["PromptContent", "SkillOutput"]].to_dict(orient="records")
    error_map: Dict[str, str] = {
        rec["PromptContent"]: rec["SkillOutput"]
        for rec in records
    }

    # 4) Assemble per-workload output
    output: List[Dict] = []
    for entry in wl_groups:
        wl = entry["workload"]
        qs = entry["questions"]
        items = []
        for q in qs:
            err = error_map.get(q)
            items.append({
                "question": q,
                "errors": [err] if err is not None else []
            })
        output.append({
            "workload": wl,
            "items": items
        })

    return output

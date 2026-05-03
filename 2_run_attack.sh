#!/bin/bash

# Run with: bash 2_run_attack.sh
# Purpose:
# 1) Run phase-2 prompt stealing/evaluation in parallel for all categories.
# 2) Write per-category logs and result/<category>_res_<target_model>.csv outputs.
# 3) Post-process results: merge into result/demo_data_all_categories_res_<target_model>.csv,
#    export stolen_prompts/stolen_prompts_<target_model>.py, then delete per-category result CSVs.

set -euo pipefail

tasks="Ads Business Code Email Ideas SEO Writing Food Health Music Data Fashion Games Language Sports Study Translate Travel"

TARGET_LLM_MODEL="${TARGET_LLM_MODEL:-CommandA}"
CUSTOM="${CUSTOM:-false}"
CUSTOM_SCENARIO="${CUSTOM_SCENARIO:-custom}"
scenario_suffix=""
custom_arg=""
if [ "$CUSTOM" = "true" ]; then
    scenario_suffix="_${CUSTOM_SCENARIO}"
    custom_arg="--custom --custom_scenario ${CUSTOM_SCENARIO}"
fi

parallel_jobs=6

output_dir="log"

mkdir -p "$output_dir"

current_jobs=0

for task in $tasks
do
    nohup venv/bin/python -u 2_run_attack.py \
        --theme "$task" \
        $custom_arg \
        --target_llm_model "$TARGET_LLM_MODEL" \
        --out "${output_dir}/${task}_${TARGET_LLM_MODEL}${scenario_suffix}.txt" \
        >> "${output_dir}/${task}_${TARGET_LLM_MODEL}${scenario_suffix}.log" 2>&1 &

    current_jobs=$((current_jobs + 1))
    if [ "$current_jobs" -ge "$parallel_jobs" ]; then
        wait
        current_jobs=0
    fi
done

wait

# Post-process per-category result files:
# 1) extract stolen prompts to a dict file
# 2) merge category results to a consolidated csv
# 3) delete each per-category result file
python3 - "$tasks" "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import csv
import os
import sys

tasks = [x for x in sys.argv[1].split() if x]
target_model = sys.argv[2]
scenario_suffix = sys.argv[3]
result_dir = "result"
merged_path = os.path.join(result_dir, f"demo_data_all_categories_res_{target_model}{scenario_suffix}.csv")
stolen_prompts_dir = "stolen_prompts"
os.makedirs(stolen_prompts_dir, exist_ok=True)
stolen_prompts_path = os.path.join(stolen_prompts_dir, f"stolen_prompts_{target_model}{scenario_suffix}.py")

rows = []
fieldnames = None
stolen_prompts = {}

for category in tasks:
    path = os.path.join(result_dir, f"{category}_res_{target_model}{scenario_suffix}.csv")
    if not os.path.exists(path):
        print(f"[Warning] Missing result file: {path}")
        continue

    first_prompt = None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = list(reader.fieldnames or [])
        for idx, row in enumerate(reader):
            row["Category"] = category
            rows.append(row)

            if idx == 0:
                first_prompt = (row.get("stolen prompt") or "").strip()

    if first_prompt:
        stolen_prompts[category] = first_prompt

    os.remove(path)

if not rows:
    print("[Warning] No result rows found to merge.")
    raise SystemExit(0)

if "Category" not in fieldnames:
    fieldnames = ["Category"] + fieldnames

os.makedirs(result_dir, exist_ok=True)
with open(merged_path, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(stolen_prompts_path, "w", encoding="utf-8") as f:
    f.write("stolen_prompts = {\n")
    for category in sorted(stolen_prompts):
        f.write(f"    {category!r}: {stolen_prompts[category]!r},\n")
    f.write("}\n")

print(f"Merged {len(rows)} rows into {merged_path}.")
print(f"Saved {len(stolen_prompts)} stolen prompts into {stolen_prompts_path}.")
PY

echo "All attack jobs executed and post-processing completed."

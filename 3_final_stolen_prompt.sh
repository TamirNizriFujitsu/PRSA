#!/bin/bash

# Run with: bash 3_final_stolen_prompt.sh
# Purpose:
# 1) Run phase-3 category ranking in parallel for all categories.
# 2) Write per-category model/diff_array_<category>_<target_model><scenario_suffix>.json.
# 3) Merge all per-category diff arrays into model/diff_array_all_<target_model><scenario_suffix>.json
#    (per-category files are kept).

set -euo pipefail

echo "phase 3 started"

tasks="Ads Business Code Email Ideas SEO Writing Food Health Music Data Fashion Games Language Sports Study Translate Travel NoCategory"

TARGET_LLM_MODEL="${TARGET_LLM_MODEL:-gpt-4o}"
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
> llm_calls.jsonl  # reset — count phase 3 calls only
mkdir -p result

EXTRACT_BEST_STOLEN_PROMPT_START=$(date +%s)

for task in $tasks
do
    nohup venv/bin/python -u 3_final_stolen_prompt.py \
        --theme "$task" \
        $custom_arg \
        --target_llm_model "$TARGET_LLM_MODEL" \
        >> "${output_dir}/${task}_phase3_${TARGET_LLM_MODEL}${scenario_suffix}.log" 2>&1 &

    current_jobs=$((current_jobs + 1))
    if [ "$current_jobs" -ge "$parallel_jobs" ]; then
        wait
        current_jobs=0
    fi
done

wait

# Merge all per-category diff arrays into one combined JSON and delete per-category files.
# File format: {<category>: {<category_1>: diff,...., <category_18>: diff}}
venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys
import utils

target_model = sys.argv[1]
scenario_suffix = sys.argv[2]

out_path = f"result/diff_array_all_{target_model}{scenario_suffix}.json"
utils.merge_phase3_diff_arrays(out_path, target_model, scenario_suffix)
PY

venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys
import utils
import csv

target_model = sys.argv[1]
scenario_suffix = sys.argv[2]
documentation_path = f"result/final_documentation_{target_model}{scenario_suffix}.txt"
ranking_array_path = f"result/diff_array_all_{target_model}{scenario_suffix}.json"

best_category_with_ranking, categories_average = utils.choose_best_stolen_prompt(ranking_array_path)
best_category = best_category_with_ranking[0]
print(f"Best stolen prompt category: {best_category}")

target_prompt_path = f"demo_data/demo_data_all_categories{scenario_suffix}.csv"
with open(target_prompt_path, newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    target_prompt = list(reader)[0]["Prompt"]

stolen_prompt_path = f"result/post_phase2_data_{target_model}{scenario_suffix}.csv"
with open(stolen_prompt_path, newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    matching_row = next((row for row in rows if row["category"] == best_category), None)
    if matching_row is None:
        raise ValueError(f"Could not find category '{best_category}' in {stolen_prompt_path}")
    stolen_prompt = matching_row["edited stolen prompt"]

asr = utils.calculate_asr(target_prompt, stolen_prompt)
print(f"ASR: {asr}")

with open(documentation_path, "a", encoding="utf-8") as f:
    f.write(f"categories_average = {categories_average}\n")
    f.write(f"best_category = {best_category_with_ranking}\n")
    f.write(f"asr = {asr}\n\n")
PY

EXTRACT_BEST_STOLEN_PROMPT_ELAPSED=$(( $(date +%s) - EXTRACT_BEST_STOLEN_PROMPT_START))
EXTRACT_BEST_STOLEN_PROMPT_MIN=$(( EXTRACT_BEST_STOLEN_PROMPT_ELAPSED / 60 ))
EXTRACT_BEST_STOLEN_PROMPT_SEC=$(( EXTRACT_BEST_STOLEN_PROMPT_ELAPSED % 60 ))

echo "phase 3 - extracting_best_stolen_prompt = ${EXTRACT_BEST_STOLEN_PROMPT_MIN}m ${EXTRACT_BEST_STOLEN_PROMPT_SEC}s" >> "result/final_documentation_${TARGET_LLM_MODEL}${scenario_suffix}.txt"

# Full pipeline LLM call summary (phases 1+2+3)
venv/bin/python3 - "$TARGET_LLM_MODEL" "$scenario_suffix" <<'PY'
import sys, utils
utils.write_llm_call_summary(sys.argv[1], sys.argv[2], label="After phase 3")
PY

echo "All phase-3 jobs executed and diff arrays merged."

echo "phase 3 ended"
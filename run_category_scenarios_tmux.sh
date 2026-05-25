#!/bin/bash

# Run with: bash run_category_scenarios_tmux.sh
# If started outside tmux, this script opens a tmux session and runs there.

set -u

session_name="${TMUX_SESSION_NAME:-prsa_category_pipeline}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${TMUX:-}" ] && [ -z "${PRSA_PIPELINE_IN_TMUX:-}" ]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not installed or not on PATH" >&2
        exit 1
    fi

    if tmux new-session -d -s "$session_name" "cd '$repo_dir' && PRSA_PIPELINE_IN_TMUX=1 bash ./run_category_scenarios_tmux.sh"; then
        echo "Started tmux session: $session_name"
        echo "Attach with: tmux attach -t $session_name"
        exit 0
    fi

    echo "Failed to start tmux session: $session_name" >&2
    exit 1
fi

cd "$repo_dir" || exit 1

# Select which spreadsheet categories to run. NoCategory is intentionally absent
# from Scenarios Per Category.xlsx, so it is not included here.
categories=(
    Code
    Health
    Food
    SEO
    Travel
    Writing
)

models=(
    gpt-4o
    cohere-command-a
)

scenario_suffix="ScenarioBestModels_Improve_Method2"
workbook="Scenarios Per Category.xlsx"
report_file="script_report"
run_log_dir="log/script_runs"

mkdir -p "$run_log_dir"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_report() {
    echo "[$(timestamp)] $*" >> "$report_file"
}

log_report "SCRIPT_START session=${session_name} workbook=${workbook}"

prompt_table="$(mktemp)"
trap 'rm -f "$prompt_table"' EXIT

python3 - "$workbook" > "$prompt_table" <<'PY'
import base64
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

path = sys.argv[1]
main_ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
rel_ns = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
office_rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def column_number(cell_ref):
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    total = 0
    for char in match.group(1):
        total = total * 26 + ord(char) - 64
    return total


def rich_text(node):
    return "".join(t.text or "" for t in node.findall(".//a:t", main_ns))


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("a:is", main_ns)
        return rich_text(inline) if inline is not None else ""

    value = cell.find("a:v", main_ns)
    if value is None:
        return ""

    text = value.text or ""
    if cell_type == "s":
        return shared_strings[int(text)]
    return text


with ZipFile(path) as archive:
    shared_strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            rich_text(item)
            for item in shared_root.findall("a:si", main_ns)
        ]

    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pr:Relationship", rel_ns)
    }

    first_sheet = workbook.find(".//a:sheet", main_ns)
    relationship_id = first_sheet.attrib[f"{{{office_rel_ns}}}id"]
    target = rel_targets[relationship_id]
    sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target

    sheet = ET.fromstring(archive.read(sheet_path))
    rows = []
    for row in sheet.findall(".//a:sheetData/a:row", main_ns):
        cells = {}
        for cell in row.findall("a:c", main_ns):
            col = column_number(cell.attrib.get("r", ""))
            cells[col] = cell_value(cell, shared_strings)
        if cells:
            max_col = max(cells)
            rows.append([cells.get(index, "") for index in range(1, max_col + 1)])

header_index = None
category_col = None
prompt_col = None
for index, row in enumerate(rows):
    normalized = [str(value).strip() for value in row]
    if "Category" in normalized and "Original System Prompt" in normalized:
        header_index = index
        category_col = normalized.index("Category")
        prompt_col = normalized.index("Original System Prompt")
        break

if header_index is None:
    raise SystemExit("Could not find Category and Original System Prompt columns")

for row in rows[header_index + 1:]:
    category = row[category_col].strip() if category_col < len(row) else ""
    prompt = row[prompt_col] if prompt_col < len(row) else ""
    if not category or not prompt:
        continue
    encoded_prompt = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    print(f"{category}\t{encoded_prompt}")
PY
if [ $? -ne 0 ]; then
    log_report "FAILED reason=workbook_parse_failed workbook=${workbook}"
    exit 1
fi

for selected_category in "${categories[@]}"; do
    matching_line="$(awk -F '\t' -v wanted="$selected_category" '$1 == wanted { print; exit }' "$prompt_table")"
    if [ -z "$matching_line" ]; then
        log_report "SKIPPED category=${selected_category} reason=not_found_in_workbook"
        continue
    fi

    category="${matching_line%%$'\t'*}"
    prompt_b64="${matching_line#*$'\t'}"
    prompt="$(printf '%s' "$prompt_b64" | base64 -d)"
    custom_scenario="${category}${scenario_suffix}"

    for model in "${models[@]}"; do
        run_log="${run_log_dir}/${category}_${model}_${custom_scenario}.log"
        expected_output="result/diff_array_all_${model}_${custom_scenario}.json"

        log_report "START category=${category} model=${model} scenario=${custom_scenario} log=${run_log}"

        if CUSTOM=true \
           CUSTOM_SCENARIO="$custom_scenario" \
           TARGET_LLM_MODEL="$model" \
           CUSTOM_TARGET_PROMPT="$prompt" \
           bash 1_prompt_attention_generation.sh > "$run_log" 2>&1; then
            if [ -f "$expected_output" ]; then
                log_report "SUCCESS category=${category} model=${model} scenario=${custom_scenario} output=${expected_output}"
            else
                log_report "FAILED category=${category} model=${model} scenario=${custom_scenario} reason=missing_output expected=${expected_output} log=${run_log}"
            fi
        else
            exit_code=$?
            log_report "FAILED category=${category} model=${model} scenario=${custom_scenario} exit_code=${exit_code} log=${run_log}"
        fi
    done
done

log_report "SCRIPT_END session=${session_name}"

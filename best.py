from collections import defaultdict
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "result" / "diff_array_all_Qwen-1.5B_SkinCare.json"

with DATA_PATH.open(encoding="utf-8") as f:
    data = json.load(f)

sums = defaultdict(float)
counts = defaultdict(int)

for outer in data.values():
    for category, value in outer.items():
        sums[category] += value
        counts[category] += 1

averages = {k: round(sums[k] / counts[k], 3) for k in sums}

sorted_averages = dict(sorted(averages.items(), key=lambda x: x[1], reverse=True))

# get best category
best = max(averages.items(), key=lambda x: x[1])

print(f"best: {best}")
print (f"average: {sorted_averages}")

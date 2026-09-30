import subprocess, json
import pandas as pd
import numpy as np

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"

# 分页读取 0 到 250 行的所有单元格
all_cells = []
for start_r in range(0, 260, 50):
    end_r = start_r + 49
    payload = {
        "file_id": "YWhOVrcmbBuI",
        "sheet_id": "BB08J2",
        "start_row": start_r,
        "end_row": end_r,
        "start_col": 0,
        "end_col": 15
    }
    cmd = ["mcporter", "call", "tencent-docs", "sheet.get_cell_data", "--args", json.dumps(payload)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(p.stdout)
        cells = data.get("cells", [])
        all_cells.extend(cells)
        print(f"Loaded rows {start_r}..{end_r}: {len(cells)} non-empty cells")
    except Exception as e:
        print(f"Error at {start_r}..{end_r}:", e)

# 组织为字典并打印结构
matrix = {}
for c in all_cells:
    r, col = c["row"], c["col"]
    val = c.get("string_value") if c.get("value_type") == "STRING" else c.get("number_value")
    if r not in matrix: matrix[r] = {}
    matrix[r][col] = val

print("\n--- Diagnostic of first 40 rows in sheet ---")
for r in sorted(matrix.keys()):
    if r <= 45 or r >= 220:
        row_dict = matrix[r]
        cols_summary = [f"c{c}:{row_dict[c]}" for c in sorted(row_dict.keys()) if str(row_dict[c]).strip()]
        print(f"Row {r:3d} | " + " | ".join(cols_summary[:6]))
PY
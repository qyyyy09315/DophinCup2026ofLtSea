import subprocess, json
import pandas as pd
import numpy as np

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"

# 1. 抓取当前腾讯表格全部行数据
print("Fetching all rows from Tencent sheet...")
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
    except Exception as e:
        print(f"Error at {start_r}:", e)

# 矩阵组织
matrix = {}
for c in all_cells:
    r, col = c["row"], c["col"]
    val = c.get("string_value") if c.get("value_type") == "STRING" else c.get("number_value")
    if r not in matrix: matrix[r] = {}
    matrix[r][col] = val

# 2. 提取真正的 ID 序列 (Row 1..32 的 col 0)
correct_ids_top = []
for r in range(1, 33):
    if r in matrix and 0 in matrix[r] and str(matrix[r][0]).strip():
        correct_ids_top.append((r, str(matrix[r][0]).strip()))

print(f"Found {len(correct_ids_top)} top IDs from row 1..32:")
print([x[1] for x in correct_ids_top[:10]])

# 3. 提取从 Row 34 开始的数据体
data_rows = []
for r in sorted(matrix.keys()):
    if r >= 34:
        row_d = matrix[r]
        # 如果该行有内容 (col 1 或 col 2 不为空)
        if any(c in row_d and str(row_d[c]).strip() for c in range(1, 13)):
            data_rows.append((r, row_d))

print(f"Found {len(data_rows)} data rows starting from row 34.")

# 检查前 10 行的数据匹配:
print("\n--- Inspecting first 10 data rows alignment ---")
for i in range(min(10, len(data_rows))):
    old_r, d = data_rows[i]
    # 正确的 ID 应该来自 correct_ids_top[i]
    correct_id = correct_ids_top[i][1] if i < len(correct_ids_top) else d.get(0, "")
    wrong_id = d.get(0, "")
    desc = d.get(1, "")
    code = d.get(9, "")
    print(f"Index {i:2d} (Old Row {old_r:3d}) -> Target ID: {correct_id:8s} | Was: {wrong_id:8s} | Desc: {desc[:30]} | Code: {code[:25]}")
PY
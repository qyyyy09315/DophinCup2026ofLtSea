import os, sys, time, json, subprocess

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")

with open(os.path.join(TMP_DIR, "clean_tencent_table.json"), encoding="utf-8") as f:
    table = json.load(f)

print(f"Total rows to write: {len(table)}")

FILE_ID = "YWhOVrcmbBuI"
SHEET_ID = "BB08J2"

# 1. 先清空 0..260 行区域的所有旧单元格内容，彻底清除幽灵行与错位残留
print("1. Clearing old 0..260 rows to eliminate ghost rows and offsets...")
clear_payload = {
    "file_id": FILE_ID,
    "sheet_id": SHEET_ID,
    "start_row": 0,
    "end_row": 260,
    "start_col": 0,
    "end_col": 15
}
cmd_clear = ["mcporter", "call", "tencent-docs", "sheet.clear_range_cells", "--args", json.dumps(clear_payload)]
p_clear = subprocess.run(cmd_clear, capture_output=True, text=True, encoding="utf-8")
print("Clear status:", "OK" if '"error": ""' in p_clear.stdout else p_clear.stdout[:200])

# 2. 构造所有待写入的单元格列表 (Row 0..193)
all_cell_items = []

for r_idx, row in enumerate(table):
    for c_idx, val in enumerate(row):
        val_str = str(val).strip()
        if not val_str or val_str == "nan":
            continue
        
        is_num = False
        try:
            num_v = float(val)
            if not any(sep in val_str for sep in ['.py', ':', '-', '，', '、', ' ']) and val_str.count('.') <= 1:
                if not (val_str.startswith('B') or val_str.startswith('C') or val_str.startswith('T') or val_str.startswith('I') or val_str.startswith('E') or val_str.startswith('R')):
                    is_num = True
        except:
            is_num = False
            
        item = {
            "row": r_idx,
            "col": c_idx,
            "value_type": "NUMBER" if is_num else "STRING"
        }
        if is_num:
            item["number_value"] = float(val)
        else:
            item["string_value"] = val_str
            
        all_cell_items.append(item)

print(f"Total non-empty cells to write: {len(all_cell_items)}")

# 3. 按每批 35 个单元格批量高速写入 (避免超 Windows 命令行参数长度)
t0 = time.time()
batch_size = 35
for i in range(0, len(all_cell_items), batch_size):
    batch = all_cell_items[i:i+batch_size]
    payload = {
        "file_id": FILE_ID,
        "sheet_id": SHEET_ID,
        "values": batch
    }
    cmd = ["mcporter", "call", "tencent-docs", "sheet.set_range_value", "--args", json.dumps(payload)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if '"error": ""' not in p.stdout:
        print(f"Error at batch {i}..{i+len(batch)}:", p.stdout[:200], p.stderr[:200])
    else:
        if (i // batch_size) % 10 == 0 or i + len(batch) >= len(all_cell_items):
            print(f"  Batch {i:4d}..{i+len(batch):4d} / {len(all_cell_items)} written OK in {time.time()-t0:.1f}s")

print(f"\nAll {len(all_cell_items)} cells across 194 rows written successfully in {time.time()-t0:.1f}s!")

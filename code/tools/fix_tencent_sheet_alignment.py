import os, sys, time, json, re, subprocess
import pandas as pd
import numpy as np

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# 1. 组合已下载的所有原始单元格
print("1. Loading raw cells from cache...")
all_cells = []
for p in [os.path.join(TMP_DIR, 'raw_0_99.json'), os.path.join(TMP_DIR, 'raw_100_199.json'), os.path.join(TMP_DIR, 'raw_200_250.json')]:
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
        all_cells.extend(d.get('cells', []))

matrix = {}
for c in all_cells:
    r, col = c['row'], c['col']
    val = c.get('string_value') if c.get('value_type') == 'STRING' else c.get('number_value')
    if r not in matrix: matrix[r] = {}
    matrix[r][col] = val

with open(os.path.join(PROJECT_ROOT, 'doc', '代码索引.md'), encoding='utf-8') as f:
    code_index_text = f.read()

file_to_id = {}
for line in code_index_text.split('\n'):
    if line.startswith('|') and '.py' in line:
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) >= 2:
            fname = parts[0]
            m = re.match(r'^([A-Za-z0-9]+)-', fname)
            if m:
                fid = m.group(1)
                file_to_id[fname] = fid
                file_to_id[fname.split('.')[0]] = fid

top_ids = [str(matrix[r].get(0, '')).strip() for r in range(1, 34)]

# 2. 备注精炼简写函数
def shorten_remark(s):
    s = str(s).strip()
    if not s or s == "0" or s == "nan": return ""
    # 规则替换
    s = s.replace("研究，", "研，").replace("负（", "研，").replace("有效（", "效，").replace("关闭；", "关；").replace("关闭", "关")
    s = re.sub(r'不重开[A-Za-z0-9/]+', '', s)
    s = re.sub(r'未超C002b 0\.2714', '未超C002b', s)
    s = re.sub(r'较C011a([+-]\d+\.\d+)', r'OOF\1', s)
    s = s.replace("，平台仍锁C011a 45.7665", "").replace("平台仍锁C011a", "")
    s = s.strip("；， ")
    # 限制最大长度 40 字符
    if len(s) > 45:
        s = s[:42] + "..."
    return s

# 3. 构造重构后的标准表格行 (Row 0..193)
# 表头
header_row = [
    "编号",
    "模型结构(+用于分割层次，-用于表示同一层)",
    "Recall",
    "Precision",
    "AUC",
    "macro-F1",
    "提交AUC",
    "提交得分",
    "备注",
    "代码文件名/使用资源的文件名1",
    "署名",
    "日期",
    "参考文献/资料（如有）"
]

reconstructed_table = [header_row]

for idx, old_r in enumerate(range(34, 227), start=1):
    old_row_dict = matrix.get(old_r, {})
    code_raw = str(old_row_dict.get(9, '')).strip()
    primary_file = code_raw.split('/')[0].strip()
    
    # 解析真实 ID
    if idx <= len(top_ids):
        true_id = top_ids[idx - 1]
    elif primary_file in file_to_id:
        true_id = file_to_id[primary_file]
    else:
        m = re.match(r'^([A-Za-z0-9]+)-', primary_file)
        true_id = m.group(1) if m else str(old_row_dict.get(0, ''))
        
    # 修正特例
    if primary_file == "E006-esm2-650-lora.py": true_id = "E006b"
    elif primary_file == "E006a-esm2-650-mah.py": true_id = "E006a"
    elif primary_file == "I057-stack.py": true_id = "I057"
    elif primary_file == "I058-calibrated.py": true_id = "I058"
    elif primary_file == "I059-transductive-diffusion.py": true_id = "I059"
    elif primary_file == "I060-plm-mlp-ensemble.py": true_id = "I060"
    
    row_values = []
    for c in range(13):
        if c == 0:
            row_values.append(true_id)
        elif c == 8: # 备注
            row_values.append(shorten_remark(old_row_dict.get(8, "")))
        else:
            val = old_row_dict.get(c, "")
            row_values.append(val)
            
    reconstructed_table.append(row_values)

print(f"Constructed clean table with {len(reconstructed_table)} rows (Header + 193 experiments).")
print(f"Last row: {reconstructed_table[-1][0]} | {reconstructed_table[-1][1][:25]} | F1={reconstructed_table[-1][5]}")

# 4. 保存为本地 JSON
with open(os.path.join(TMP_DIR, "clean_tencent_table.json"), "w", encoding="utf-8") as f:
    json.dump(reconstructed_table, f, ensure_ascii=False, indent=2)

print(f"Saved clean table structure to tmp/clean_tencent_table.json")
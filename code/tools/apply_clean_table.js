const fs = require("fs");
const path = require("path");
const { execSync, execFileSync } = require('child_process');

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const TMP_DIR = path.join(PROJECT_ROOT, "tmp");

const table = JSON.parse(fs.readFileSync(path.join(TMP_DIR, "clean_tencent_table.json"), "utf-8"));
console.log(`Loaded ${table.length} rows from clean table JSON`);

const FILE_ID = "YWhOVrcmbBuI";
const SHEET_ID = "BB08J2";

// 1. 准备有效单元格列表 (Row 0..193)
const allCells = [];
for (let r = 0; r < table.length; r++) {
  const row = table[r];
  for (let c = 0; c < 13; c++) {
    const val = row[c];
    const strVal = (val === "" || val === null || val === undefined) ? "" : String(val).trim();

    let isNum = false;
    const num = Number(strVal);
    if (strVal && !isNaN(num) && !/[a-zA-Z:\-，、\s]/.test(strVal) && (strVal.match(/\./g) || []).length <= 1) {
      isNum = true;
    }

    if (isNum) {
      allCells.push({ row: r, col: c, value_type: "NUMBER", number_value: num });
    } else {
      allCells.push({ row: r, col: c, value_type: "STRING", string_value: strVal });
    }
  }
}

// 2. 为旧的冗余行 194..230 准备清空单元格 (置为空字符串)
for (let r = table.length; r <= 230; r++) {
  for (let c = 0; c < 13; c++) {
    allCells.push({ row: r, col: c, value_type: "STRING", string_value: "" });
  }
}

console.log(`Total cells to update/clear: ${allCells.length}`);

// 3. 批量高速写入
const batchSize = 40;
const t0 = Date.now();
const tempArgFile = path.join(TMP_DIR, "batch_arg.json").replace(/\\/g, "/");

for (let i = 0; i < allCells.length; i += batchSize) {
  const batch = allCells.slice(i, i + batchSize);
  const payload = {
    file_id: FILE_ID,
    sheet_id: SHEET_ID,
    values: batch,
  };
  fs.writeFileSync(tempArgFile, JSON.stringify(payload), "utf-8");
  try {
    const argsJson = fs.readFileSync(tempArgFile, "utf-8");
    execFileSync("mcporter", ["call", "tencent-docs", "sheet.set_range_value", "--args", argsJson], { shell: false });
    if (Math.floor(i / batchSize) % 15 === 0 || i + batch.length >= allCells.length) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`  Updated ${i + batch.length} / ${allCells.length} cells in ${elapsed}s`);
    }
  } catch (err) {
    console.error(`Error at batch ${i}:`, err.message);
  }
}

console.log(`\nAll 194 rows and trailing rows updated cleanly!`);


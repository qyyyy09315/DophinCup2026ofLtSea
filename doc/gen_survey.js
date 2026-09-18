const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, PageNumber, AlignmentType, HeadingLevel, WidthType,
  BorderStyle, ShadingType, VerticalAlign, NumberFormat, SectionType,
  TableOfContents, PageBreak, TableLayoutType,
} = require("docx");
const fs = require("fs");
const path = require("path");

const SUN = { ascii: "Times New Roman", eastAsia: "SimSun" };
const HEI = { ascii: "Times New Roman", eastAsia: "SimHei" };
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = {
  top: NB, bottom: NB, left: NB, right: NB,
  insideHorizontal: NB, insideVertical: NB,
};
const BODY = 28;
const REF = 24;
const LINE = 360;
const IND = 560;
const pg = { width: 11906, height: 16838 };
const bodyMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417, header: 850, footer: 992 };

function t(text, opt) {
  return new TextRun(Object.assign({ text, font: SUN, size: BODY, color: "000000" }, opt || {}));
}
function cite() {
  const nums = Array.prototype.slice.call(arguments);
  return new TextRun({
    text: "[" + nums.join(",") + "]",
    superScript: true,
    size: 21,
    font: { ascii: "Times New Roman" },
    color: "000000",
  });
}
function p(children) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: IND },
    spacing: { line: LINE, after: 0, before: 0 },
    children,
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 280, line: LINE },
    children: [new TextRun({ text, bold: true, size: 32, font: HEI, color: "000000" })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 360, after: 160, line: LINE },
    keepNext: true,
    children: [new TextRun({ text, bold: true, size: 30, font: HEI, color: "000000" })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 240, after: 120, line: LINE },
    keepNext: true,
    children: [new TextRun({ text, bold: true, size: 28, font: HEI, color: "000000" })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 200, after: 80, line: 312 },
    children: [new TextRun({ text, font: SUN, size: 21, color: "000000" })],
  });
}
function note(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 40, after: 200, line: 312 },
    children: [new TextRun({ text, font: SUN, size: 21, color: "000000" })],
  });
}
function refItem(text) {
  return new Paragraph({
    indent: { left: 480, hanging: 480 },
    spacing: { line: 360, after: 80 },
    children: [new TextRun({ text, font: SUN, size: REF, color: "000000" })],
  });
}
function buildHeader(title) {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000" } },
      spacing: { after: 60 },
      children: [new TextRun({ text: title, size: 18, color: "333333", font: SUN })],
    })],
  });
}
function buildPageNumberFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: "- ", size: 21, font: { ascii: "Times New Roman" } }),
        new TextRun({ children: [PageNumber.CURRENT], size: 21, font: { ascii: "Times New Roman" } }),
        new TextRun({ text: " -", size: 21, font: { ascii: "Times New Roman" } }),
      ],
    })],
  });
}
function threeLine(headers, rows, colW) {
  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((h, i) => new TableCell({
      width: { size: colW[i], type: WidthType.PERCENTAGE },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 12, color: "000000" },
        bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000" },
        left: NB, right: NB,
      },
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: h, bold: true, size: 21, font: HEI, color: "000000" })],
      })],
    })),
  });
  const dataRows = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((cell, i) => new TableCell({
      width: { size: colW[i], type: WidthType.PERCENTAGE },
      borders: {
        top: NB, left: NB, right: NB,
        bottom: ri === rows.length - 1
          ? { style: BorderStyle.SINGLE, size: 12, color: "000000" }
          : NB,
      },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [new TextRun({ text: cell, size: 21, font: SUN, color: "000000" })],
      })],
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 12, color: "000000" },
      bottom: { style: BorderStyle.SINGLE, size: 12, color: "000000" },
      left: NB, right: NB, insideHorizontal: NB, insideVertical: NB,
    },
    rows: [headerRow].concat(dataRows),
  });
}

function estimateTextWidth(text, pt) {
  let width = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0);
    const isCJK = (code >= 0x4E00 && code <= 0x9FFF) || (code >= 0x3400 && code <= 0x4DBF) ||
      (code >= 0x3000 && code <= 0x303F) || (code >= 0xFF00 && code <= 0xFFEF) ||
      (code >= 0x2E80 && code <= 0x2EFF);
    width += isCJK ? pt * 20 : pt * 11;
  }
  return width;
}
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."，。、；：！？的与和及之在于为-_—–·/ \t"]);
  const lines = [];
  let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++) {
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
      }
    }
    if (breakAt === -1) {
      breakAt = charsPerLine;
      const prevChar = remaining[breakAt - 1];
      const nextChar = remaining[breakAt];
      if (prevChar && nextChar && !breakAfter.has(prevChar) && !breakAfter.has(nextChar) &&
          /[\u4e00-\u9fff]/.test(prevChar) && /[\u4e00-\u9fff]/.test(nextChar)) {
        breakAt = breakAt - 1;
      }
    }
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}
function calcTitleLayoutMixed(title, maxWidthTwips, preferredPt, minPt) {
  let titlePt = preferredPt;
  let lines;
  while (titlePt >= minPt) {
    const cpl = Math.max(2, Math.floor(maxWidthTwips / (titlePt * 20)));
    lines = splitTitleLines(title, cpl);
    const tooWide = lines.some((l) => estimateTextWidth(l, titlePt) > maxWidthTwips);
    if (lines.length <= 3 && !tooWide) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    const cpl = Math.max(2, Math.floor(maxWidthTwips / (minPt * 20)));
    lines = splitTitleLines(title, cpl);
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}
function calcR5MetaLayout(metaEntries, fontPt) {
  const maxLabelLen = Math.max(...metaEntries.map((e) => [...e.label].length));
  const labelNeedTw = (maxLabelLen + 2) * fontPt * 20;
  const valueNeedTw = 5000;
  const totalNeedTw = labelNeedTw + valueNeedTw;
  const tablePct = Math.min(75, Math.max(55, Math.ceil(totalNeedTw / 11906 * 100)));
  const rawLabelPct = Math.ceil(labelNeedTw / (tablePct / 100 * 11906) * 100);
  return { tablePct, labelPct: Math.max(25, Math.min(45, rawLabelPct)) };
}
function buildR5MetaTable(metaEntries) {
  const { tablePct, labelPct } = calcR5MetaLayout(metaEntries, 12);
  const valuePct = 100 - labelPct;
  const bottomBorder = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const rows = metaEntries.map((entry) => new TableRow({
    cantSplit: true,
    children: [
      new TableCell({
        width: { size: labelPct, type: WidthType.PERCENTAGE },
        borders: noBorders,
        margins: { top: 60, bottom: 60, left: 0, right: 0 },
        children: [new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: { before: 60, after: 60, line: 400 },
          children: [new TextRun({ text: entry.label + "：", size: 24, font: SUN, color: "000000" })],
        })],
      }),
      new TableCell({
        width: { size: valuePct, type: WidthType.PERCENTAGE },
        borders: { top: NB, left: NB, right: NB, bottom: bottomBorder },
        margins: { top: 60, bottom: 60, left: 80, right: 0 },
        children: [new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: { before: 60, after: 60, line: 400 },
          children: [new TextRun({ text: entry.value, size: 24, font: SUN, color: "000000" })],
        })],
      }),
    ],
  }));
  return new Table({
    width: { size: tablePct, type: WidthType.PERCENTAGE },
    alignment: AlignmentType.CENTER,
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows,
  });
}
function buildCoverR5(config) {
  const PAGE_H = 16838, SAFETY = 1200;
  const safeH = PAGE_H - SAFETY;
  const simMarginLR = 1701, simMarginT = 1200;
  const contentW = 11906 - simMarginLR * 2;
  const { titlePt, titleLines } = calcTitleLayoutMixed(config.title, contentW, 28, 22);
  const titleSize = titlePt * 2;
  const metaEntries = (config.metaLines || []).map((line) => {
    const sep = line.indexOf("：") !== -1 ? "：" : ":";
    const idx = line.indexOf(sep);
    if (idx === -1) return { label: line, value: "" };
    return { label: line.slice(0, idx).trim(), value: line.slice(idx + sep.length).trim() };
  });
  const schoolNameH = config.schoolName ? (22 * 23 + 400) : 0;
  const titleTotalH = titleLines.length * (titlePt * 23 + 200);
  const subtitleH = config.subtitle ? (15 * 23 + 600) : 0;
  const metaTableH = metaEntries.length * 520;
  const footerH = config.footerRight ? (12 * 23 + 200) : 0;
  const spacerParas = 3 * 350;
  const fixedH = schoolNameH + titleTotalH + subtitleH + metaTableH + footerH + spacerParas;
  const remaining = Math.max(safeH - fixedH, 600);
  const topSpacing = Math.min(Math.floor(remaining * 0.28) + simMarginT, 4200);
  const midSpacing = Math.min(Math.floor((remaining - simMarginT) * 0.18), 2000);
  const bottomSpacing = Math.min(remaining - topSpacing + simMarginT - midSpacing, 5500);
  const children = [];
  children.push(new Paragraph({ spacing: { before: topSpacing } }));
  if (config.schoolName) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400, line: Math.ceil(22 * 23), lineRule: "atLeast" },
      children: [new TextRun({
        text: config.schoolName, size: 44, characterSpacing: 40, color: "000000", font: SUN,
      })],
    }));
  }
  if (config.docType) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 300, line: Math.ceil(16 * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: config.docType, size: 32, color: "000000", font: HEI })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: {
        after: i < titleLines.length - 1 ? 120 : 300,
        line: Math.ceil(titlePt * 23),
        lineRule: "atLeast",
      },
      children: [new TextRun({
        text: titleLines[i], size: titleSize, bold: true, color: "000000", font: HEI,
      })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200, line: Math.ceil(15 * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: config.subtitle, size: 30, italics: true, color: "000000", font: SUN })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: midSpacing } }));
  if (metaEntries.length > 0) children.push(buildR5MetaTable(metaEntries));
  children.push(new Paragraph({ spacing: { before: bottomSpacing } }));
  if (config.footerRight) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: config.footerRight, size: 24, color: "404040", font: SUN })],
    }));
  }
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: PAGE_H, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: "FFFFFF" },
        borders: noBorders,
        verticalAlign: VerticalAlign.TOP,
        margins: { left: simMarginLR, right: simMarginLR },
        children,
      })],
    })],
  })];
}

const refs = [
  "[1] ASHBURNER M, BALL C A, BLAKE J A, et al. Gene ontology: tool for the unification of biology[J]. Nature Genetics, 2000, 25(1): 25-29.",
  "[2] THE GENE ONTOLOGY CONSORTIUM. The Gene Ontology resource: enriching a GOld mine[J]. Nucleic Acids Research, 2021, 49(D1): D325-D334.",
  "[3] THE UNIPROT CONSORTIUM. UniProt: the Universal Protein Knowledgebase in 2023[J]. Nucleic Acids Research, 2023, 51(D1): D523-D531.",
  "[4] RADIVOJAC P, CLARK W T, ORON T R, et al. A large-scale evaluation of computational protein function prediction[J]. Nature Methods, 2013, 10(3): 221-227.",
  "[5] ZHOU N, JIANG Y, BERGQUIST T R, et al. The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens[J]. Genome Biology, 2019, 20: 244.",
  "[6] ALTSCHUL S F, MADDEN T L, SCHAFFER A A, et al. Gapped BLAST and PSI-BLAST: a new generation of protein database search programs[J]. Nucleic Acids Research, 1997, 25(17): 3389-3402.",
  "[7] BUCHFINK B, XIE C, HUSON D H. Fast and sensitive protein alignment using DIAMOND[J]. Nature Methods, 2015, 12(1): 59-60.",
  "[8] BUCHFINK B, REUTER K, DROST H G. Sensitive protein alignments at tree-of-life scale using DIAMOND[J]. Nature Methods, 2021, 18(4): 366-368.",
  "[9] STEINEGGER M, SODING J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets[J]. Nature Biotechnology, 2017, 35(11): 1026-1028.",
  "[10] TORONEN P, MEDLAR A, HOLM L. PANNZER2: a rapid functional annotation web server[J]. Nucleic Acids Research, 2018, 46(W1): W84-W88.",
  "[11] YOU R, ZHANG Z, XIONG Y, et al. GOLabeler: improving sequence-based large-scale protein function prediction by learning to rank[J]. Bioinformatics, 2018, 34(14): 2465-2473.",
  "[12] YAO S, YOU R, WANG S, et al. NetGO 2.0: improving large-scale protein function prediction with massive sequence, text, domain, family and network information[J]. Nucleic Acids Research, 2021, 49(W1): W358-W366.",
  "[13] WANG S, YOU R, LIU Y, et al. NetGO 3.0: protein language model improves large-scale functional annotations[J]. Genomics, Proteomics & Bioinformatics, 2023, 21(2): 349-358.",
  "[14] KULMANOV M, HOEHNDORF R. DeepGOPlus: improved protein function prediction from sequence[J]. Bioinformatics, 2020, 36(2): 422-429.",
  "[15] SANDERSON T, BILESCHI M L, BELANGER D, et al. ProteInfer, deep neural networks for protein functional inference[J]. eLife, 2023, 12: e80942.",
  "[16] YU F, KOLTUN V. Multi-scale context aggregation by dilated convolutions[C]//International Conference on Learning Representations. 2016.",
  "[17] RIDNIK T, BEN-BARUCH E, ZAMIR N, et al. Asymmetric loss for multi-label classification[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 82-91.",
  "[18] ELNAGGAR A, HEINZINGER M, DALLAGO C, et al. ProtTrans: toward understanding the language of life through self-supervised learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, 44(10): 7112-7127.",
  "[19] LIN Z, AKIN H, RAO R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model[J]. Science, 2023, 379(6637): 1123-1130.",
  "[20] LIU Q, ZHANG C, FREDDOLINO L. InterLabelGO+: unraveling label correlations in protein function prediction[J]. Bioinformatics, 2024, 40(11): btae655.",
  "[21] YUAN Q, XIE J, XIE J, et al. Fast and accurate protein function prediction from sequence through pretrained language model and homology-based label diffusion[J]. Briefings in Bioinformatics, 2023, 24(3): bbad117.",
  "[22] HEINZINGER M, WEISSENOW K, SANCHEZ J G, et al. ProstT5: bilingual language model for protein sequence and structure[J]. NAR Genomics and Bioinformatics, 2024, 6(4): lqae150.",
  "[23] LIU W, WANG Z, YOU R, et al. PLMSearch: protein language model powers accurate and fast sequence search for remote homology[J]. Nature Communications, 2024, 15: 2775.",
  "[24] HONG L, HU Z, SUN S, et al. Fast, sensitive detection of protein homologs using deep dense retrieval[J]. Nature Biotechnology, 2025, 43: 983-995.",
  "[25] SU J, HE Y, YOU S, et al. A trimodal protein language model enables advanced protein searches[J]. Nature Biotechnology, 2025. DOI: 10.1038/s41587-025-02836-0.",
  "[26] CHERVOV A, VAKHRUSHEV A, FIRONOV S, et al. ProtBoost: protein function prediction with Py-Boost and graph neural networks: CAFA5 top-2 solution[EB/OL]. (2024-12-05)[2026-09-10]. https://arxiv.org/abs/2412.04529.",
  "[27] ZHANG L, WANG Y, CHEN X, et al. GOBoost: leveraging long-tail gene ontology terms for accurate protein function prediction[J]. Bioinformatics, 2025, 41(6): btaf267.",
  "[28] CORMACK G V, CLARKE C L A, BUETTCHER S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods[C]//Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval. New York: ACM, 2009: 758-759.",
  "[29] VAN DER LAAN M J, POLLEY E C, HUBBARD A E. Super learner[J]. Statistical Applications in Genetics and Molecular Biology, 2007, 6(1): 25.",
  "[30] CHOU K C. Prediction of protein cellular attributes using pseudo-amino acid composition[J]. Proteins, 2001, 43(3): 246-255.",
  "[31] CAO Y, SHEN Y. TALE: Transformer-based protein function Annotation with joint sequence-Label Embedding[J]. Bioinformatics, 2021. DOI: 10.1093/bioinformatics/btab198.",
  "[32] ZHANG C, FREDDOLINO P L. A large-scale assessment of sequence database search tools for homology-based protein function prediction[EB/OL]. (2023)[2026-09-10]. https://doi.org/10.1101/2023.11.14.567021.",
  "[33] IBTEHAZ N, KAGAYA Y, KIHARA D. Domain-PFP allows protein function prediction using function-aware domain embedding representations[J]. Communications Biology, 2023, 6: 1103.",
  "[34] KAMINSKI K, LUDWICZAK J, PAWLICKI K, et al. pLM-BLAST: distant homology detection based on direct comparison of sequence representations from protein language models[J]. Bioinformatics, 2023, 39(10): btad579.",
  "[35] YANG Y. A study of thresholding strategies for text categorization[C]//Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. New York: ACM, 2001: 137-145.",
  "[36] WU J, QING H, OUYANG J, et al. HiFun: homology independent protein function prediction by a novel protein-language self-attention model[J]. Briefings in Bioinformatics, 2023, 24(5): bbad311.",
  "[37] ZENG S, WANG D, XU D. PEFT-SP: parameter-efficient fine-tuning on large protein language models improves signal peptide prediction[EB/OL]. (2023)[2026-09-10]. https://doi.org/10.1101/2023.11.04.565642.",
  "[38] HOU C, LIU D, SHEN Y. Unsupervised discovery of functional sequence patterns from protein language model with MotifAE[J]. Nature Communications, 2026. DOI: 10.1038/s41467-026-77333-2.",
  "[39] HOU C, LIU D, ZAFAR A, et al. Understanding language model scaling on protein fitness prediction[J]. Nature Computational Science, 2026. DOI: 10.1038/s43588-026-01010-z.",
  "[40] CHEN B, CHENG X, LI P, et al. xTrimoPGLM: unified 100-billion-parameter pretrained transformer for deciphering the language of proteins[J]. Nature Methods, 2025, 22: 1028-1039.",
  "[41] BOADU F, WANG Y, CHENG J. A unified multimodal model for generalizable zero-shot and supervised protein function prediction[J]. Bioinformatics, 2026. DOI: 10.1093/bioinformatics/btag356.",
  "[42] CUTURI M. Sinkhorn distances: lightspeed computation of optimal transport[C]//Advances in Neural Information Processing Systems 26. 2013.",
  "[43] JUMPER J, EVANS R, PRITZEL A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.",
  "[44] BARUTCUOGLU Z, SCHAPIRE R E, TROYANSKAYA O G. Hierarchical multi-label prediction of gene function[J]. Bioinformatics, 2006, 22(7): 830-836.",
  "[45] FAN H, YAN W, WANG L, et al. Deep learning-based multi-functional therapeutic peptides prediction with a multi-label focal dice loss function[J]. Bioinformatics, 2023, 39(6): btad334.",
  "[46] CHEN Z, LUO Q. DualNetGO: a dual network model for protein function prediction via effective feature selection[J]. Bioinformatics, 2024, 40(8): btae437.",
  "[47] REIJNDERS M J M F. Wei2GO: weighted sequence similarity-based protein function prediction[J]. PeerJ, 2022, 10: e12931.",
  "[48] HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.",
  "[49] CERRI R, BARROS R C, DE CARVALHO A C, et al. Reduction strategies for hierarchical multi-label classification in protein function prediction[J]. BMC Bioinformatics, 2016, 17: 373.",
  "[50] FRIEDBERG I. Automated protein function prediction: the genomic challenge[J]. Briefings in Bioinformatics, 2006, 7(3): 225-242.",
  "[51] CHOU K C. Using amphiphilic pseudo amino acid composition to predict enzyme subfamily classes[J]. Bioinformatics, 2005, 21(1): 10-19.",
  "[52] CHEN T, GUESTRIN C. XGBoost: a scalable tree boosting system[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, 2016: 785-794.",
  "[53] ZHANG H, CISSE M, DAUPHIN Y N, et al. mixup: beyond empirical risk minimization[C]//International Conference on Learning Representations. 2018.",
  "[54] CHARTE F, RIVERA A J, DEL JESUS M J, et al. MLSMOTE: approaching imbalanced multilabel learning through synthetic instance generation[J]. Knowledge-Based Systems, 2015, 89: 385-397.",
  "[55] CHEN Z M, WEI X S, WANG P, et al. Multi-label image recognition with graph convolutional networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 5177-5186.",
  "[56] RIDNIK T, SHARIR G, BEN-COHEN A, et al. ML-Decoder: scalable and versatile classification head[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023: 32-41.",
  "[57] LI M, SHI W, ZHANG F, et al. A deep learning framework for predicting protein functions with co-occurrence of GO terms[J]. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2023, 20(2): 1058-1068.",
  "[58] MANFREDI M, SAVOJARDO C, MARTELLI P L, et al. E-SNPs&GO: embedding of protein sequence and function improves the annotation of human pathogenic variants[J]. Bioinformatics, 2022, 38(16): 4059-4061.",
  "[59] ZHANG Z, LU J, CHENTHAMARAKSHAN V, et al. ProtIR: iterative refinement between retrievers and predictors for protein function annotation[EB/OL]. (2024)[2026-09-10]. https://arxiv.org/abs/2402.07955.",
  "[60] SU J, LU Y, PAN S, et al. RoFormer: enhanced transformer with rotary position embedding[J]. Neurocomputing, 2024, 568: 127063.",
];

const coverChildren = buildCoverR5({
  schoolName: "【单位全称】",
  docType: "综　述",
  title: "基于序列的蛋白质功能预测方法研究进展",
  subtitle: "Advances in Sequence-Based Protein Function Prediction",
  metaLines: [
    "作者：【作者姓名】",
    "单位：【学院或实验室】",
    "通信作者：【姓名】",
    "收稿日期：2026-09-10",
  ],
  footerRight: "参考文献采用 GB/T 7714 顺序编码制",
});

const front = [];
front.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 200, line: LINE },
  children: [new TextRun({ text: "摘　要", bold: true, size: 32, font: HEI, color: "000000" })],
}));
front.push(p([
  t("蛋白质功能注释是连接序列积累与生物学机制理解的关键环节。随着高通量测序使蛋白质序列规模持续扩大，实验测定功能的速度远滞后于序列增长，计算预测因此成为注释体系的重要补充。本文围绕仅依赖氨基酸序列的功能预测问题，按方法假设梳理同源转移、序列深度学习、蛋白质语言模型以及多源融合与后处理四条主线，并对照CAFA等社区评测所揭示的共识与分歧。现有证据表明：在近缘同源充分时，比对与k-mer邻域仍提供最稳定的标签转移信号；在远同源与无命中情形下，卷积神经网络与预训练语言模型构成主要兜底；将序列模型与同源证据在推理阶段融合，通常优于单一路径。与此同时，基因本体层级、蛋白质\u2014蛋白质相互作用与结构信息虽可提升公开基准表现，却对标签匿名、仅序列设定构成约束。本文进一步讨论远同源检索、长尾标签、阈值策略与评测口径等开放问题，以期为后续方法设计提供可对照的文献框架。"),
]));
front.push(new Paragraph({
  spacing: { before: 160, after: 200, line: LINE },
  children: [
    new TextRun({ text: "关键词：", bold: true, size: BODY, font: HEI, color: "000000" }),
    t("蛋白质功能预测；同源检索；卷积神经网络；蛋白质语言模型；多标签学习；CAFA"),
  ],
}));
front.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  alignment: AlignmentType.CENTER,
  spacing: { before: 280, after: 200, line: LINE },
  children: [new TextRun({ text: "Abstract", bold: true, size: 32, font: { ascii: "Times New Roman" }, color: "000000" })],
}));
front.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: IND },
  spacing: { line: LINE },
  children: [t("Protein function annotation remains a bottleneck between the rapid accumulation of amino-acid sequences and mechanistic understanding in biology. Because experimental characterization cannot keep pace with sequencing, computational prediction has become an indispensable complement. This review organizes sequence-only function prediction along four methodological lines: homology transfer, sequence deep learning, protein language models, and multi-source fusion with post-processing, and contrasts community consensus with unresolved debates under CAFA-style evaluation. Current evidence indicates that alignment and k-mer neighborhoods remain the most reliable vehicles of label transfer when close homologs are available, whereas convolutional networks and pretrained language models provide the principal fallback for remote or homology-free cases. Combining sequence models with homology evidence at inference typically outperforms either source alone. Gene Ontology hierarchy, interaction networks and structure can improve public benchmarks, yet they are unavailable under anonymous-label, sequence-only constraints. Open issues include remote-homology retrieval, long-tail labels, decision thresholds and evaluation protocols.")],
}));
front.push(new Paragraph({
  spacing: { before: 160, after: 360, line: LINE },
  children: [
    new TextRun({ text: "Keywords: ", bold: true, size: BODY, font: { ascii: "Times New Roman" }, color: "000000" }),
    t("protein function prediction; homology search; convolutional neural network; protein language model; multi-label learning; CAFA"),
  ],
}));

const toc = [];
toc.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 280 },
  children: [new TextRun({ text: "目　录", bold: true, size: 32, font: HEI, color: "000000" })],
}));
toc.push(new TableOfContents("Table of Contents", {
  hyperlink: true,
  headingStyleRange: "1-3",
}));
toc.push(new Paragraph({
  spacing: { before: 200 },
  children: [new TextRun({
    text: "说明：目录由域代码生成。编辑完成后请在目录上右键选择\u201c更新域\u201d，以刷新页码。",
    italics: true, size: 18, color: "888888", font: SUN,
  })],
}));

const body = [];
body.push(h1("1　引言"));
body.push(p([
  t("蛋白质执行催化、信号转导、结构支撑与分子识别等多样功能。基因本体（Gene Ontology, GO）将功能表述为分子功能、生物学过程与细胞组分三类受控词表，并借助有向无环图刻画术语间的层级关系"),
  cite(1, 2),
  t("。UniProt等知识库持续汇集实验与推断注释"),
  cite(3),
  t("，但未表征序列的增长速度仍显著高于高质量标注。因此，由氨基酸序列推断功能标签，长期构成生物信息学的核心任务之一"),
  cite(4, 5, 50),
  t("。"),
]));
body.push(p([
  t("该任务在形式上清点为多标签分类：一条序列可同时对应多个功能术语，正例稀疏、标签相关，评测需兼顾召回与精确。社区范围的CAFA评测表明，计算方法能够产生可被实验验证的新注释，同时揭示不同本体分支与物种上的表现并不均衡"),
  cite(4, 5),
  t("。早期方法以序列比对转移同源蛋白的已知功能为主；随后出现基于组成特征与浅层学习器的统计模型；近十年则形成以卷积神经网络、蛋白质语言模型及多组件学习排序为代表的深度方法谱系"),
  cite(11, 14, 18, 19),
  t("。"),
]));
body.push(p([
  t("需要强调的是，公开文献中的\u201c功能预测\u201d并不对应单一问题设定。多数工作默认可使用GO层级、结构预测、相互作用网络或文献文本；另一些竞赛与应用场景则仅提供氨基酸序列，且标签被匿名化为固定维数的指示向量。若将依赖本体图或外部注释库的方法直接迁移到后一类设定，其增益往往无法复现。本文因此以序列为主要输入，按\u201c证据从何而来、在何种条件下有效\u201d组织文献，而不是按发表年份逐篇摘录。"),
]));
body.push(p([
  t("覆盖范围需要事先说明。本文主要依据Nature、Science、Nature Methods、Nature Biotechnology、Bioinformatics、Nucleic Acids Research及CAFA系列评测报告中与序列功能预测直接相关的工作，并补充若干近年预印本中已被后续正式发表或被社区广泛引用的方法。该范围足以支撑方法路线的对照，但并不构成对该领域全部工作的穷尽普查；结构组学、代谢网络与单细胞功能推断等相邻方向仅在与序列预测接口处提及。"),
]));
body.push(p([
  t("全文安排如下。第2节给出任务形式与评测口径。第3节综述同源与组成特征路径。第4节讨论序列卷积神经网络。第5节讨论蛋白质语言模型及其下游适应。第6节讨论融合、排序与后处理。第7节讨论远同源、长尾标签与结构\u2014文本多模态。第8节归纳共识、局限与可检验的后续方向。"),
]));

body.push(h1("2　任务形式、数据与评测"));
body.push(h2("2.1　问题形式化"));
body.push(p([
  t("给定长度为L的氨基酸序列x，预测器输出在标签集合Y上的得分向量s(x)，再经阈值或截断规则得到二值注释。当Y为GO术语时，其规模可达数万，并受\u201c子术语蕴含父术语\u201d的层级约束"),
  cite(1, 44),
  t("。当Y为竞赛中的匿名标签时，层级与术语文本均不可用，模型只能依赖序列与训练集中的共现统计。两种设定共享多标签不平衡，却对可利用的先验结构截然不同。"),
]));
body.push(p([
  t("输入侧同样存在分叉。同源方法需要可检索的已注释序列库；结构方法需要实验结构或AlphaFold一类预测结构"),
  cite(43),
  t("；网络方法需要蛋白质相互作用或功能关联图"),
  cite(12, 46),
  t("；文本方法需要功能描述或文献"),
  cite(41),
  t("。仅序列方法则把上述信息压缩进预训练权重，或完全放弃。评测结论必须与输入假设绑定，否则容易高估方法的可迁移性。"),
]));
body.push(h2("2.2　数据资源与注释证据"));
body.push(p([
  t("GO与UniProt构成当前最常用的监督来源"),
  cite(2, 3),
  t("。注释质量并不均匀：实验证据码与电子推断证据码混杂，时间分割评测正是为了减少\u201c用未来注释训练今日模型\u201d的泄漏"),
  cite(4),
  t("。结构域数据库、酶分类号与文献摘要常作为辅助组件进入GOLabeler与NetGO一类系统"),
  cite(11, 12),
  t("。若应用禁止外部注释库，则这些组件不能作为推理期输入，最多只能通过在允许语料上预训练的参数间接出现。"),
]));
body.push(h2("2.3　评测协议"));
body.push(p([
  t("CAFA以时间分割模拟\u201c今日未知、他日被实验确认\u201d的真实场景，主要报告Fmax以及宏平均精确\u2014召回曲线下面积，并按分子功能、生物学过程与细胞组分分别统计"),
  cite(4, 5),
  t("。Fmax在决策阈值上取最优F1，有利于比较排序质量，但并不等同于固定阈值下的提交得分。部分竞赛采用宏平均F1并对每一标签等权，从而放大稀有标签与低基数样本的影响。阈值策略本身即构成独立研究问题：逐标签SCut、按列分位PCut与按样本基数RCut在文本分类中已有系统比较"),
  cite(35),
  t("，迁移到蛋白质功能时，\u201c最优阈值在验证集上搜得\u201d与\u201c测试分布漂移\u201d之间的张力仍然存在。"),
]));
body.push(p([
  t("此外，命中同源与未命中同源的样本应分开报告。比对拷贝在高一致度子集上即可达到可观的F1，真正的方法差异往往集中在远同源与无命中子集。若只报告全局均值，容易把同源覆盖率的提升误读为表示学习的进步。"),
]));

body.push(h1("3　同源转移与组成特征"));
body.push(h2("3.1　比对与快速同源检索"));
body.push(p([
  t("若查询序列与已注释蛋白共享可检测的进化关系，则将邻居标签按比对质量加权聚合，是迄今最稳健的功能推断路径之一。BLAST与PSI-BLAST奠定了基于局部比对的检索范式"),
  cite(6),
  t("。面向宏基因组与超大规模蛋白库，DIAMOND以双重索引显著提高通量，并在后续版本中把灵敏度推进到接近BLAST的水平"),
  cite(7, 8),
  t("。MMseqs2则在聚类与迭代剖面搜索上提供可扩展实现"),
  cite(9),
  t("。系统比较表明，默认参数下不同检索引擎的命中集合并不重合，敏感度档与E值阈值会改变召回，但不一定改善最终的宏平均F1"),
  cite(32),
  t("。"),
]));
body.push(p([
  t("在注释系统层面，PANNZER2等服务器把加权同源、描述统计与GO证据码组织为可部署流程"),
  cite(10),
  t("。Wei2GO强调以bitscore非线性加权抑制低质量邻居"),
  cite(47),
  t("。这类方法的共同前提是：邻居标签在功能上可转移。当一致度进入所谓的\u201c暮光区\u201d，错误转移与漏检会同时上升，单纯放宽E值往往引入噪声。"),
]));
body.push(h2("3.2　k-mer与组成描述子"));
body.push(p([
  t("在比对未命中时，短k-mer共享仍可能捕获局部模体。DeepGOPlus将CNN扫描与序列相似性预测线性组合，使无结构、无网络的序列方法达到当时CAFA3前列"),
  cite(14),
  t("。其相似性分支本质上是把注释库中的k-mer重叠转化为标签投票，与DIAMOND邻域互补而非重复。Chou提出的伪氨基酸组成把疏水、亲水相关滞后引入固定维特征"),
  cite(30, 51),
  t("，在酶亚家族等小规模任务上有效，但在高维多标签且已有强同源路的情形下，边际信息有限。"),
]));
body.push(p([
  t("结构域级表示提供了另一条\u201c局部化\u201d思路。Domain-PFP以功能感知的结构域嵌入降低全序列平均带来的信号稀释"),
  cite(33),
  t("。若应用场景缺少InterPro分区，以滑动窗口k-mer近似结构域，是工程上的折中，其有效性取决于窗口是否与真实功能单元对齐。"),
]));

body.push(h1("4　序列卷积神经网络"));
body.push(h2("4.1　从模体扫描到残差膨胀栈"));
body.push(p([
  t("卷积神经网络把功能预测视为沿序列轴的模体检测。DeepGOPlus以一维卷积捕获局部模式，再与相似性分支融合"),
  cite(14),
  t("。ProteInfer采用更深的残差膨胀卷积，直接从全长序列预测GO与酶分类，并报告与BLAST学到的信息并不重合，简单集成即可超过单路"),
  cite(15),
  t("。膨胀卷积在不降低分辨率的前提下扩大感受野"),
  cite(16),
  t("，残差连接则缓解深层一维网络的优化困难"),
  cite(48),
  t("。实践中，浅层多尺度分支有时优于盲目加深：通道数、窗口长度与正则会显著改变校准，而校准误差会在随后的阈值搜索中被放大。"),
]));
body.push(h2("4.2　多标签损失与类别不平衡"));
body.push(p([
  t("功能注释正例远少于负例。非对称损失对易分负例降权并裁剪可能的误标负例，在通用多标签图像基准上取得过显著增益"),
  cite(17),
  t("，也被移植到蛋白质CNN训练。焦点Dice等复合损失同样针对正负不平衡与标签共现"),
  cite(45),
  t("。合成少数类与输入混合则提供数据侧补救：MLSMOTE面向多标签少数类生成"),
  cite(54),
  t("，Mixup以凸组合平滑决策边界"),
  cite(53),
  t("。需要区分的是：损失或增强改善的是固定阈值下的校准，还是排序质量。若评测采用逐标签搜阈值的F1，表面增益可能部分被阈值吸收。InterLabelGO+在训练目标中显式加入标签依赖，并与比对动态加权"),
  cite(20),
  t("；若仅替换损失而不改变表示与融合，经验上往往不足以形成可重复的全局增益。"),
]));
body.push(p([
  t("标签共现图卷积与查询式分类头提供了另一条建模相关的路径"),
  cite(55, 56, 57),
  t("。它们在图像多标签与GO共现设定中理由充分，但当标签被匿名、共现矩阵由有限训练集估计时，图传播容易把频率先验写成预测，从而在宏平均下伤害稀有标签。"),
]));

body.push(h1("5　蛋白质语言模型及其下游适应"));
body.push(h2("5.1　预训练表示"));
body.push(p([
  t("蛋白质语言模型在大规模未标注序列上做掩码或自回归预训练，使残基表示编码进化与结构相关信息。ProtTrans系统评估了Transformer在蛋白质语料上的迁移"),
  cite(18),
  t("。ESM-2进一步表明，仅凭单序列语言模型即可达到可与多序列比对方法竞争的结构预测精度"),
  cite(19),
  t("。ProstT5在序列与三维邻接词表之间做双语训练，意在把结构信号写入表示"),
  cite(22),
  t("。旋转位置编码等位置偏置也被引入蛋白序列建模，以改善长度外推"),
  cite(60),
  t("。百亿至千亿参数的统一模型被报告用于更广的蛋白质理解任务"),
  cite(40),
  t("，但其下游功能分类收益并不随参数单调上升：适应度预测中已观察到过大规模导致似然过冲"),
  cite(39),
  t("。"),
]));
body.push(h2("5.2　读出、微调与冻结嵌入"));
body.push(p([
  t("将语言模型用于功能预测，至少有三条适配路径。其一，提取均值或分类标记嵌入，再接线性头、梯度提升或k近邻"),
  cite(21, 26, 52),
  t("。该路径计算量低，但均值池化会在长序列上稀释局部分子识别信号，嵌入空间也可能过度聚集，使近邻失去区分度。其二，参数高效微调，例如LoRA，在信号肽等任务上已显示优于仅训练探针"),
  cite(37),
  t("。InterLabelGO+取ESM-2末若干层残基特征训练神经网络，再与DIAMOND动态加权，在CAFA5中取得竞争性名次"),
  cite(20),
  t("。其三，残基级注意力池化或查询池化，试图在不引入接触图的前提下保留位点信息。SPROF-GO将注意力池化与同源标签扩散相结合"),
  cite(21),
  t("。HiFun则强调在无同源时依赖自注意力序列模型"),
  cite(36),
  t("。TALE一类工作进一步把标签本身嵌入同一空间，使序列\u2014术语对齐成为可学习目标"),
  cite(31),
  t("。"),
]));
body.push(p([
  t("一个反复出现的经验是：冻结嵌入上的线性探针或k近邻，不能代表微调后的语言模型上限；反之，微调失败也不能直接推广为\u201c该骨干无效\u201d，因为读出头、损失与序列截断都是强混淆因素。稀疏自编码器被用于从语言模型中拆解可解释模体"),
  cite(38),
  t("，目前更接近分析工具而非可替换的分类头。把公开GO分类头当作\u201c本地注释器\u201d再映射到匿名标签，则把外部注释压缩进了权重，并不改变其对外部监督的依赖。"),
]));

body.push(h1("6　融合、排序与决策后处理"));
body.push(h2("6.1　组件学习与集成"));
body.push(p([
  t("GOLabeler把序列、结构域、文献与网络等组件得分纳入学习排序框架，在大规模GO预测上优于单组件"),
  cite(11),
  t("。NetGO系列进一步引入海量网络信息，并在3.0中加入蛋白质语言模型"),
  cite(12, 13),
  t("。ProtBoost以多目标梯度提升拟合语言模型嵌入，再用图网络按GO层级做堆叠，在CAFA5中名列前茅"),
  cite(26),
  t("。这些工作的共同点是：单路很少最优，增益来自异质证据的互补。其代价是依赖可对齐的标签空间与组件校准。当标签匿名、无法使用GO图时，层级堆叠无法直接调用，只能退化为在冻结分路上学习凸组合。超学习器理论为交叉验证选择基学习器权重提供了统计依据"),
  cite(29),
  t("，但若基学习器已在同一验证协议下高度相关，再学习组合权容易过拟合局部校准，而不增加新信息。"),
]));
body.push(h2("6.2　秩融合、最优传输与阈值"));
body.push(p([
  t("当不同来源的分数不可比时，互惠秩融合以1/(k+rank)相加，避免显式校准"),
  cite(28),
  t("。该方法对排序敏感、对幅度不敏感，因此常与保留幅度的线性混合并用。Sinkhorn迭代提供另一种把行基数与列频率同时对齐的软匹配"),
  cite(42),
  t("，其温度与边际松弛需要在验证协议上锁定，否则容易把先验频率写成预测。决策阶段，逐标签阈值、样本级截断与分层阈值各有适用区间"),
  cite(35),
  t("：在同源命中子集上截断假阳性，有时比在全局上移动阈值更安全；在无命中子集上降低阈值以提高召回，则可能被宏平均中的精确率惩罚抵消。"),
]));
body.push(p([
  t("检索器与预测器的迭代精化试图在推理期互相校正邻居与标签"),
  cite(59),
  t("。该思路在可更新检索库时有吸引力；若检索库仅含训练序列且首轮预测已经过校准，二次重加权的增量通常很小，并可能放大首轮误差。"),
]));

body.push(h1("7　远同源、长尾标签与多模态扩展"));
body.push(h2("7.1　语言模型驱动的远同源检索"));
body.push(p([
  t("传统比对在低一致度下召回下降。pLM-BLAST直接比较残基级语言模型表示以检测远同源"),
  cite(34),
  t("。PLMSearch表明，以语言模型嵌入做密集检索，可在速度与远同源灵敏度上超过常规序列搜索"),
  cite(23),
  t("。DHR将对比学习与双编码器检索结合，在结构分类基准上提高家族级灵敏度，并可与剖面方法级联以构造多序列比对"),
  cite(24),
  t("。上述方法为\u201c无BLAST命中\u201d样本提供了新的邻居来源。需要注意的是，检索到的邻居若用于标签转移，仍取决于嵌入是否编码功能而非单纯的长度或组成偏差；以冻结均值嵌入直接替换比对邻域，可能只是重复已有的k-mer信号。"),
]));
body.push(h2("7.2　标签层级与长尾"));
body.push(p([
  t("GO的层级使父术语频率远高于叶术语。早期工作以贝叶斯网络施加父\u2014子一致性"),
  cite(44),
  t("，后续有层次多标签的归约策略"),
  cite(49),
  t("。GOBoost针对长尾GO项构造标签图与头尾分模型"),
  cite(27),
  t("。当标签被匿名且频数并不极端时，为长尾GO设计的图传播或焦点损失，并不自动对应\u201c低基数样本\u201d问题：前者是术语频率，后者是每条蛋白的正标签个数，二者不可互换。"),
]));
body.push(h2("7.3　结构、网络与文本"));
body.push(p([
  t("AlphaFold使结构几乎可对任意序列取得"),
  cite(43),
  t("，因而结构感知词表与结构\u2014序列融合成为近期热点。三模态检索模型试图在序列、结构与功能文本之间建立统一索引"),
  cite(25),
  t("。零样本工作将序列嵌入与功能描述对齐，以便在未见术语上泛化"),
  cite(41),
  t("。DualNetGO等则显式引入网络与特征选择"),
  cite(46),
  t("。这些方向在拥有GO名称、结构文件或相互作用的基准上理由充分；在禁止外部注释库、标签无文本的设定下，它们首先遇到的是接口缺失，而不是优化不足。变异注释中把序列嵌入与功能描述联合编码的工作，同样说明文本通道的价值依赖于可读的功能词表"),
  cite(58),
  t("。"),
]));

body.push(h1("8　结论与展望"));
body.push(p([
  t("综合现有文献，可以审慎地归纳三点共识。第一，同源证据在可检测时仍然主导功能转移，快速比对与k-mer邻域是这一路径的工程实现，而不是被深度学习替代的遗留模块"),
  cite(7, 14, 32),
  t("。第二，序列深度模型与蛋白质语言模型的价值，主要体现在比对失效的区域，其读出方式与损失至少与骨干同等重要"),
  cite(15, 19, 20),
  t("。第三，异质证据的融合在公开GO基准上反复有效，但融合公式必须与可用先验匹配：有层级时可用图堆叠与学习排序，无层级时则应避免把共现传播写成必然增益"),
  cite(11, 13, 26),
  t("。"),
]));
body.push(p([
  t("相应的开放问题同样具体。如何在不引入外部注释库的前提下，把密集远同源检索转化为可校准的标签转移，仍缺少与CAFA口径对齐的充分消融。如何在宏平均F1下同时改善无命中样本与低基数样本，而不伤害高一致度子集，需要把切片指标写入标准报告。如何避免把GO文本、结构词表与相互作用图的方法，误判为仅序列问题的可选项，则是评测设计与应用约束需要共同澄清的问题。未来若要形成可积累的进展，宜在固定数据切分与固定决策规则下，一次只改变一条证据路径，并以独立对照与融合对照同时报告。"),
]));

body.push(caption("表1　主要方法路线的输入假设与适用条件"));
body.push(threeLine(
  ["路线", "核心证据", "典型方法", "主要依赖", "薄弱环节"],
  [
    ["同源转移", "邻居标签", "BLAST/DIAMOND/MMseqs2", "已注释序列库", "无命中、暮光区"],
    ["组成与k-mer", "局部重叠", "DeepGOPlus相似性支", "k-mer谱", "长程依赖"],
    ["序列CNN", "模体检测", "DeepGOPlus/ProteInfer", "标注序列", "远同源、校准"],
    ["语言模型", "进化表示", "ESM-2/InterLabelGO+", "预训练语料", "池化稀释、显存"],
    ["融合排序", "多源得分", "GOLabeler/NetGO", "可对齐组件", "标签空间不一致"],
  ],
  [16, 16, 24, 22, 22]
));
body.push(note("注：表中\u201c依赖\u201d指方法成立所必需的数据或先验，而非实现细节。"));

body.push(caption("表2　代表性工作的问题设定对照"));
body.push(threeLine(
  ["工作", "主要输入", "标签空间", "融合策略", "评测场景"],
  [
    ["DeepGOPlus", "序列+k-mer", "GO", "CNN与相似性线性加", "CAFA3"],
    ["ProteInfer", "全长序列", "GO/EC", "与BLAST后验集成", "公开基准"],
    ["GOLabeler/NetGO", "多组件", "GO", "学习排序", "CAFA"],
    ["InterLabelGO+", "ESM-2+比对", "GO", "动态加权", "CAFA5"],
    ["ProtBoost", "PLM嵌入+GO图", "GO", "提升+GCN堆叠", "CAFA5"],
    ["DHR/PLMSearch", "序列嵌入", "家族/检索", "密集检索", "远同源基准"],
  ],
  [18, 18, 14, 22, 18]
));
body.push(note("注：表中评测场景按原文报告口径列出，不同基准的数值不可直接横向比较。"));

body.push(h1("参考文献"));
refs.forEach((r) => body.push(refItem(r)));
body.push(new Paragraph({
  spacing: { before: 400, line: LINE },
  children: [new TextRun({
    text: "收稿日期：【年-月-日】；修回日期：【年-月-日】",
    size: 21, font: SUN, color: "000000",
  })],
}));
body.push(new Paragraph({
  spacing: { line: LINE },
  children: [new TextRun({
    text: "基金项目：【基金项目名称及编号】",
    size: 21, font: SUN, color: "000000",
  })],
}));
body.push(new Paragraph({
  spacing: { line: LINE },
  children: [new TextRun({
    text: "通信作者：【姓名】，E-mail：【邮箱】",
    size: 21, font: SUN, color: "000000",
  })],
}));

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: SUN, size: BODY, color: "000000" },
        paragraph: { spacing: { line: LINE } },
      },
      heading1: {
        run: { font: HEI, size: 32, bold: true, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 480, after: 280, line: LINE } },
      },
      heading2: {
        run: { font: HEI, size: 30, bold: true, color: "000000" },
        paragraph: { spacing: { before: 360, after: 160, line: LINE } },
      },
      heading3: {
        run: { font: HEI, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 240, after: 120, line: LINE } },
      },
    },
  },
  sections: [
    {
      properties: {
        page: { size: pg, margin: { top: 0, bottom: 0, left: 0, right: 0 } },
      },
      children: coverChildren,
    },
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: {
          size: pg,
          margin: bodyMargin,
          pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN },
        },
      },
      headers: { default: buildHeader("基于序列的蛋白质功能预测方法研究进展") },
      footers: { default: buildPageNumberFooter() },
      children: front,
    },
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: {
          size: pg,
          margin: bodyMargin,
          pageNumbers: { formatType: NumberFormat.UPPER_ROMAN },
        },
      },
      headers: { default: buildHeader("基于序列的蛋白质功能预测方法研究进展") },
      footers: { default: buildPageNumberFooter() },
      children: toc,
    },
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: {
          size: pg,
          margin: bodyMargin,
          pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
        },
      },
      headers: { default: buildHeader("基于序列的蛋白质功能预测方法研究进展") },
      footers: { default: buildPageNumberFooter() },
      children: body,
    },
  ],
});

const out = path.join(__dirname, "基于序列的蛋白质功能预测方法综述.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
}).catch((e) => {
  console.error(e);
  process.exit(1);
});

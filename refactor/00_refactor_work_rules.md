# vLLM-Omni Refactor 講解／寫作規則

本文件由 `example/example1.md`、`example/example2.md`、`example/example3.md` 抽出，作為 `refactor/` 分析文件嘅工作規則。  
用途：**講解「點寫」**；具體領域內容見對應 analysis（例如 `omni_coordinator_analysis.md`）。

---

## 1. 目的同邊界

1. 文件層級係 **analysis / RFC**，唔係最終 implementation design。
2. Target path、draft class／function、mermaid 分層圖，一律標明 **draft only**——用嚟幫 reviewer 睇 ownership 邊界，唔係承諾最終目錄名。
3. 唔好喺 analysis 度 silently 改行為；若問題牽涉行為變更，要寫清楚「reorganization」vs「behavior change」。
4. Out-of-scope 但會影響本區嘅檔案，另開一節（學 example3 嘅 *OUT of the Scope Directory but Relevant*），唔好 silently 膨脹 scope。

---

## 2. Front matter（每份 analysis 必備）

| 欄位 | 要求 |
|------|------|
| Title | 建議 `[Analysis] <area> …`（對齊 example2） |
| Disclaimer | 開頭聲明：analysis-level；proposed paths／draft code 非最終設計 |
| Area | Primary area + Primary file + Adjacent files |
| Proposed Priority | 例如 P0／P1，附一句 reason |
| Owner | GitHub handle 或 TBD |
| Code base | Branch + Commit SHA + Commit title（對齊 example1） |

可選但強烈建議：

- **Goal / Scope**（example3）：Focus bullets + 目錄清單 + rename／delete 意圖
- **Potential Path Impact Summary**（example2）：PR → current path(s) → target path(s, draft) → main problem

---

## 3. 必寫現況

順序固定：

1. **Current Responsibility** — 而家呢個檔／套件實際擁有咩（唔係「應該擁有咩」）
2. **Main Code Paths** — startup → runtime → shutdown；可用 ASCII 或 mermaid
3. **Related files** — 主檔 + 測試 + design doc（如有）
4. （可選）**File inventory** — 粗略行數，方便後面估 PR LOC

路徑圖規則：

- 標清入口（CLI／HTTP／engine）同關鍵物件名
- 標 file path（必要時加行號）
- 多路徑時用表對照（學 example1 嘅 Path／Request object 表）

---

## 4. 問題寫法（Problems Observed）

每個 Problem 必須有呢四段（可改名，但資訊唔好缺）：

1. **Current evidence** — path:line 或精簡 snippet（唔好淨係口號）
2. **Draft direction** — draft-only 介面／目錄／偽代碼
3. **How this helps** — 對 review、測試、ownership 有咩具體好處
4. **Relevant PR** — 對應邊個 PR（可多個）

可選：

- **Coordinator / PIC**（example1）— 邊個跟進
- **Why now** — 同其他 RFC（例如 stage rename）嘅依賴關係

問題選擇準則：

- 優先寫 **ownership 分裂、雙通道、god-file、可測性缺口、跨 modality／跨 family 依賴**
- 唔好堆純 naming bike-shed；rename 要服務於 ownership 或刪冗余
- AI 生成、過長 if-else、命名不清亦可以寫，但要附 evidence（example1 Problem 2）

---

## 5. PR 拆法

1. **高風險組裝面先 PR0 guardrails**（route map、app-state、membership／LB 不變式、關鍵 golden behavior）。純加測試、唔搬邏輯。
2. 之後每個 PR 只搬 **一個 ownership 邊界**（一個 family／一個合併／一層 startup）。
3. 每份 analysis 要有：
   - **Potential Path Impact Summary** 表
   - **PR Breakdown**（focus files）
   - （可選）粗略 +/- LOC；強調 reorganization、logic 行數大致持平（example2）
4. 標明同相鄰 RFC 嘅銜接（例如 Coordinator PR2 ↔ Stage RFC PR2），避免兩邊各自改同一介面。

禁止：

- 一個大 PR 同時 rename + merge + 行為變更 + 刪測試
- 未有 guardrail 就拆 public serving／distributed membership 熱路徑

---

## 6. 邊界同依賴規則

1. 寫清 **dependency direction**：邊啲套件可以 import 邊啲；邊啲永遠唔好互相 import serving／client 層。
2. Cross-boundary 服務呼叫用 **Protocol／注入**，唔好 module-level 互相 import（example2 `ImageChatBridge` 精神）。
3. Split 原則要講清楚係 **endpoint-family / ownership**，定係 pure modality／pure rename——唔好混稱。
4. 若文件畫嘅 mermaid 同正文目標有張力，**以正文合併／刪除意圖為準**，並喺文內點名（避免 reviewer 跟錯圖）。

---

## 7. Draft interface 寫法

1. 只展示 **public surface** 同關鍵建構參數，唔要貼成個檔。
2. 標 `# draft only` 或說明 “Just a draft, not the final design”。
3. Delete／rename 用清晰箭頭：`old.py -> delete`、`a.py -> b.py`。
4. 合併類（例如 Master + Coordinator）要列：
   - 接收邊啲舊參數
   - 對外保留邊啲 hub／master API
   - TCP vs IPC（或同等 transport）規則

---

## 8. 語言同交付位置

| 文件 | 語言 | 位置 |
|------|------|------|
| 講解規則（本檔） | 繁體中文 | `refactor/00_refactor_work_rules.md` |
| 可上 GitHub 嘅 analysis／RFC | 英文（對齊 example1–3） | `refactor/<area>_analysis.md` |
| 閱讀用對照版（可選） | 繁體中文 | `refactor/<area>_analysis.zh.md` |
| 參考樣例 | 英文 | `example/example1.md` … `example3.md` |

講解建議順序：

1. 先用本檔講「點寫／點拆 PR」
2. 再用對應 analysis 講「呢個 area 寫咗啲咩問題同目標」

---

## 9. 完成檢查清單（寫完 analysis 自檢）

- [ ] 已 pin branch + commit
- [ ] Area／Adjacent／Owner／Priority 齊
- [ ] 有 Current Responsibility + Main Code Paths
- [ ] 每個 Problem 有 evidence → draft → helps → PR
- [ ] 有 PR0（或說明點解唔需要）同後續細 PR
- [ ] Target path／interface 標 draft only
- [ ] Out-of-scope-but-relevant 已分隔
- [ ] 同相鄰 RFC 嘅銜接已寫一句
- [ ] 無承諾未驗證嘅行為／精度數字

---

## 10. 三份 example 對照（速查）

| 能力 | example1 | example2 | example3 |
|------|----------|----------|----------|
| Pin commit | ✓ | （issue 風格） | （RFC 風格） |
| 路徑圖 | ✓ ASCII | ✓ 步驟 | ✓ mermaid layers |
| Problem + snippet | ✓ | ✓ | 改為 per-file proposed changes |
| PR 路徑表 + LOC | | ✓ | ✓ breakdown |
| Rename／delete 清單 | | | ✓ |
| Out-of-scope relevant | | 部分 | ✓ |
| Dependency direction | | ✓ | 部分（介面草圖） |

**本專案預設混合模板**：example2 骨架 + example1 evidence 密度 + example3 Goal／rename／delete／PR Breakdown。

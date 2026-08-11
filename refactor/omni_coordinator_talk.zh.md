# OmniCoordinator 講稿（工程師場）

配合投影（**以 analysis 為準**；講稿只係口述節奏）：

| 文件 | 用途 |
|------|------|
| [`omni_coordinator_analysis.zh.md`](omni_coordinator_analysis.zh.md)／[`.md`](omni_coordinator_analysis.md) | 對外系統分析（簡體／英文）；架構、分級、PR |
| [`omni_coordinator_code_audit.zh.md`](omni_coordinator_code_audit.zh.md) | 公開 API＋逐 function 審計 |

語氣：講**碼上事實**；分開「已證實」同「草案」。對外 analysis **唔講**內部討論過程（例如方案代號）。  
建議時長：**約 15–20 分鐘**＋ Q&A。

---

## 0. 開場（1 分鐘）

今日主題：`vllm_omni/distributed/omni_coordinator` **全套件**（7 個檔）——先講清邊個做咩，再答**係咪單點**（analysis §2），再分級／PR。

**三句定調（建議照念；由淺入深）：**

1. **唔係每次 serve 都起 Coord**——只有 `DistStageRuntime`（`--stage-id` + master）。普通 in-process multi-stage **冇** Coord。（analysis §1 開首）  
2. **第一條大問題：Coord 係咪單點？**——一個子進程＋記憶體 registry、無 HA（細節：**講稿 §3／analysis §2**）。  
3. **Refactor 主線**係 **合併 Master 入 Coord**（P1；PR2）——同單點係兩件事；單點講完先再講「合併救唔救 SPOF」。

今日要聽眾帶走：

- Master 同 Coord 邊個做咩  
- 點解話 P0 單點；緩解同真 HA 點分  
- 可靠性縫＋合併主線；交 PR 時邊條係主交付  

---

## 1. 架構：邊個喺邊（3 分鐘）

（投影 analysis §1.1 結構圖＋組件表）

### 1.1 組件一句（對齊 analysis §1.1 表）

| 組件 | 一句 |
|------|------|
| `OmniCoordinatorRuntime` | 啟停 Coord；**只露** router／pub 地址 |
| `OmniCoordinator` | 記憶體 registry + ROUTER／PUB |
| `OmniMasterServer` | 註冊時派握手／I／O 地址，並回傳 router_addr |
| `OmniCoordClientForHub` | SUB 缓存；Mem 持有；Mem／Pool 讀 snapshot |
| `StagePool` + `LoadBalancer` | 選副本（LB 歸屬見 §3 P2） |
| `OmniCoordClientForStage` | update／heartbeat → Coord |

### 1.2 兩個地址（analysis §1.2）

| 地址 | 邊個 bind | 邊個用 |
|------|-----------|--------|
| `router_addr` | **Coord** ROUTER | Master **轉發字串**；Stage DEALER |
| `pub_addr` | **Coord** PUB | Head：Mem → Hub SUB |

handshake／input／output **唔經** Coord；Coord 只維護 membership。  
→ Runtime 報嘅地址 **即** Coord bind 嘅 socket。

### 1.3 Master ≠ Coord（必講清楚）

| | Master | Coord |
|--|--------|-------|
| 時機 | register **一次**（bootstrap） | 持續 update／heartbeat／PUB |
| 做乜 | 派握手／資料面 ZMQ 地址（handshake／input／output）＋ echo router | membership／活死人／queue 視圖 |
| 唔做乜 | 唔管 heartbeat、唔做 LB／pick | 唔派 handshake／IO port |

現場金句（注意：**pick 唔喺 Coord 入面做**）：

> Master 負責「點樣入場」；Coord 負責「入場之後邊個仲活」。  
> **Pick** 係 Head 側 `StagePool`＋`LoadBalancer` 讀 Hub snapshot。

### 1.4 `OmniCoordClientForStage` 邊個 new？

**唔係 Master。** Master 只回 `coordinator_router_address`（analysis 回覆欄位表）。  
Stage 進程自己起：LLM 用 `create_stage_coord_client`；Diffusion 直接 ctor（P2 S3 雙路徑）。

---

## 2. 時序（3 分鐘）

（投影 analysis §1.2／§1.3）

### 2.1 建立（現状）

口頭對齊 sequenceDiagram，唔使讀完表：

1. Head：`OmniCoordinatorRuntime` → child `OmniCoordinator` ready（router／pub）  
2. Head：起 `OmniMasterServer`（帶 router 字串）  
3. Head init／等 Stage（本地或 headless）  
4. Stage／launch → Master：**register** → 回 HS／IO／replica_id／router  
5. Head bind HS／IO；engine connect  
6. Stage 起 client → Coord：**DEALER update**  
7. Head：Mem＋Hub SUB pub → 之後可 pick  

強調：**Head 先起齊 Master＋Coord**，先至 launch／等 replica；register 攞匯合地址。

本地 vs headless（analysis 路徑表）：本地由 Head launch helper register、Head **bind** HS；headless 自行 register、**connect**。

### 2.2 使用：三條線（唔好講錯因果）

| 線 | 邊個 | 做乜（對齊 analysis §1.3） |
|----|------|---------------------------|
| A | Stage client | DEALER heartbeat；Coord 改記憶體；**queue 變或 ERROR→UP** 先 schedule 廣播 |
| B | Coord periodic | timeout＋合併後 PUB；Hub 收 cache |
| C | Pool＋LB | 讀 Hub snapshot → `select` |

金句：

> **唔係每次 heartbeat 即 PUB**；pick 同單次 heartbeat **無直接因果**。

### 2.3 動態 headless（analysis §1.3 末）

服務拉起後可以再加 headless：Master `on_register` → Orchestrator → `MembershipController` attach → Pool。  
**有手動動態掛上；冇 autoscaler。**

---

## 3. P0：單點？（2 分鐘）

（投影 analysis §2）

**結論：係 availability SPOF。**

證據：

- 每個 head **一個** Coord 子進程  
- registry **只在記憶體**  
- **無** HA／failover  

掛咗之後：

- membership／LB 熱路徑冇備援；`StagePool.pick` 失敗或空轉  
- HTTP serve **未必**即死；Master handshake **仍可**獨立  

**單點講清之後，先好講呢句（開場唔好提早丟；analysis §2 緩解表）：**

> **合併 ≠ 解決單點**——清嘅係 ownership；記憶體 registry、無 HA 仍然係 P0。  
> **Watchdog ≠ HA**——只係緩解；registry 仍空，要可靠重註冊（R1／R2）；快照另論。

| 手段 | 效果 |
|------|------|
| Watchdog 重拉 Coord | 進程可自愈；registry 仍空 |
| 持久化／冷啟動快照 | 縮短空窗；仍係單活 |
| 合併 Master（PR2） | 清 ownership；**仍然可以係**單點 |
| 真 HA | 另案；本 refactor **唔做** |

現場問句（可留白）：

> 若我哋 kill Coord 子進程，pool 預期點樣？——應用測試鎖住答案（PR1）。

---

## 4. 分級：兩個 P1（3 分鐘）

（投影 analysis §3）

| 優先 | 類別 | 講法 |
|------|------|------|
| **P0** | 可用性 | 只有一個 Coord（記憶體 registry、無 HA） |
| **P1** | **可靠性** | 註冊／心跳／PUB／關閉靜默失敗或狀態唔齊（R1–R5） |
| **P1** | **重構：合併 Master** | 雙 owner 必須收束——**本任務主線 → PR2** |
| **P2** | 套件結構 | LB 歸屬、死 API、雙路徑、自訂 LB |

分級金句（analysis）：

> P0＝只有一個 Coord；P1 可靠性＝路徑唔夠可靠；P1 重構＝Master／Coord 雙 owner 必須收束。

### 4.1 P1 可靠性（R1–R5）

| # | 一句 |
|---|------|
| R1 | Stage registration 遇 `zmq.Again` **靜默 drop** → 永不進 registry |
| R2 | 未知 `input_addr` 嘅 heartbeat **直接忽略** → 首次 update 丟就永久缺席 |
| R3 | PUB／send `NOBLOCK` best-effort → Hub／Pool 視圖陳舊 |
| R4 | Runtime `terminate`／`kill`；子進程唔走 `coordinator.close()`（見 analysis §5 close 矩陣） |
| R5 | `queue_length` parse 未強制 `int` |

三者疊加（R1＋R2＋R3）：cold start／Coord 重啟後，Stage「以為註冊咗」，Hub／Pool 睇唔到。

### 4.2 P1 重構（M1／M2；主線多講）

| # | 痛點 | 方向 |
|---|------|------|
| M1 | Master 同 Coord **並列** | 併入 Coord；register／update 同一 owner |
| M2 | Hub＋Mem 間接層 | Pool **直讀** Coord（隨合併落地） |

目標形狀（投影 analysis §6 目標圖）：

- **刪除**獨立 Master／Hub／`MembershipController`（唔係「收窄」）  
- Coord 兼做 register（派 HS／IO）＋ registry  
- `StagePool` **直讀** Coord；class **沿用舊名**  

金句：

> 兩個 P1 同級；**refactor 真正要交嘅係合併 Master（PR2）**。

### 4.3 P2（對齊 analysis S1–S5）

- **S1 LB 歸屬**：純 `select`；擁有者 Head／`StagePool`；放喺套件係結構債  
- **S2 死 API**：`add_new_replica` 等；stage `update_info` 僅 tests  
- **S3** LLM／Diffusion 註冊雙路徑  
- **S4** fork／spawn 進程模型 TODO（`runtime.py`）  
- **S5 自訂 LB**：PR2；`--omni-lb-policy=mypkg.lb:MyBalancer`（或 `mypkg:MyBalancer`）

---

## 5. Code audit 速覽（2–3 分鐘）

（翻 audit §0 公開 API → §1 廢碼表；必要時 analysis §4 檔案一覽、§5 close）

### 5.1 公開面要細

`OmniCoordinatorRuntime` **對外只有**：`__init__(host, heartbeat_timeout)`、`router_address`、`pub_address`、`close`。

`OmniCoordinator` 公開 mutator（`add_new_replica` 等）**生產唔用**——走 `_handle_event`。

### 5.2 可清理 vs 必須修

**可清理（P2／PR1）：** 死公開 mutator；`update_info` 僅 tests；可收窄 `get_active_replicas` 等。  

**必須修（P1 可靠性／PR1）：** R1–R5（Again drop、未知 addr HB、PUB／SLA、優雅 close、`queue_length` int）。

---

## 6. 兩個 PR（2 分鐘）

（投影 analysis §6 總表＋PR2 目標圖）

| | **PR1** | **PR2（主線）** |
|--|---------|-----------------|
| 對應 | P0 預期＋**P1 可靠性**＋P2 小清理 | **P1 重構**＋自訂 LB |
| 做 | SPOF／註冊測試；R1–R5；死 API；**統一 LLM／Diff 路徑**；LB 歸屬（仍只內建三策略） | Master→Coord；**刪除** Hub／Mem；Pool 直讀；dotted-path 自訂 LB |
| 唔做 | 合併 Master；自訂 LB；刪 Hub／Mem | 完整 HA |
| 順序 | **建議先**（鎖行為、降回歸） | refactor **主交付**；依賴 PR1 先落地 |

金句（analysis）：

> Refactor 主線係 PR2。PR1 建議先合。**PR2 唔消除單點本身。**

### PR2 自訂 LB（口頭 20 秒）

```bash
vllm serve ... --omni-lb-policy=mypkg.lb:MyBalancer
```

字串傳到 `_build_load_balancer_factory` 先 `importlib`；內建 `random`／`round-robin`／`least-queue-length` 保留。  
上游 core **冇**呢個跨 stage `LoadBalancer`；風格近 production-stack dotted callback。

### PR2 目標時序（對照現状／analysis §6）

- 只起 Coord（含舊 Master 職責）→ Stage **向 Coord register** → client update → Pool **直讀** Coord  
- 使用期：heartbeat／registry／pick 三線，**唔經 Hub**  

---

## 7. 收束（1 分鐘）

帶走三點：

1. **Coord 係 membership／LB 熱路徑 SPOF**；serve 未必即死；watchdog≠HA  
2. **兩個 P1**：可靠性縫 ＋ **合併 Master（refactor 主線）**  
3. 落地：**PR1（建議）→ PR2（主交付）**；今日講分析，唔強制落 code  

投影收尾：analysis §3 分級表 ＋ §6 PR 表。

---

## 附：可能 Q&A

**Q：單進程 multi-stage 有冇 Coord？**  
A：冇。普通 `StageRuntime` 唔起；只有 `DistStageRuntime`。

**Q：Coord 掛會唔會令整個 vllm serve crash？**  
A：唔保證即 crash；distributed membership／pick 會壞。Master 仍可能握手。

**Q：加 watchdog 算唔算解決 P0？**  
A：算緩解。registry 仍空，要可靠重註冊。真 HA 另案。（analysis §2）

**Q：合併 Master 之後仲係唔係單點？**  
A：係。PR2 清 ownership，唔消除單點。

**Q：`OmniCoordClientForStage` 係咪 Master new？**  
A：唔係。Master 只回 router 字串；Stage 自己起。

**Q：Runtime 公開 API 係咪好多？**  
A：唔多。對外就建構＋兩個地址＋`close`。

**Q：vLLM 上游有冇同樣嘅 LoadBalancer？**  
A：core 冇跨 stage replica 呢個 class；DP 有內建 queue／prefix LB。跨 engine 分流喺 production-stack。Omni 自訂用 dotted path。

**Q：刪 dead API 會唔會破 test？**  
A：可能要改 test；生產唔依賴呢啲 public mutator。

**Q：自訂 LB 同合併 Master 可唔可以拆開？**  
A：而家一齊放 PR2。太大可拆 PR2a 合併、PR2b dotted-path LB。

**Q：動態擴縮容有冇？**  
A：有手動 headless 動態掛上；冇 autoscaler。（analysis §1.3）

---

## 附：投影檢查清單（對齊 analysis 章節）

- [ ] §1.1 現状結構圖＋組件表  
- [ ] §1.2 建立時序（含地址／回覆欄位）  
- [ ] §1.3 使用三線＋動態 headless  
- [ ] §2 P0＋緩解階梯  
- [ ] §3 兩個 P1＋R／M／S  
- [ ] §5 close 矩陣（若問 shutdown）  
- [ ] §6 PR1／PR2＋PR2 目標結構／時序  
- [ ] audit §0／§1（若問 API／廢碼）  

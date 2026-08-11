# OmniCoordinator Code Audit（逐 function）

> 配套系統分析：[`omni_coordinator_analysis.zh.md`](omni_coordinator_analysis.zh.md)／[`omni_coordinator_analysis.md`](omni_coordinator_analysis.md)  
> 講稿：[`omni_coordinator_talk.zh.md`](omni_coordinator_talk.zh.md)  
> 範圍：`vllm_omni/distributed/omni_coordinator/` 全部 7 檔  
> 基準：`main` @ `c27623c2`  
> 判定：`keep`｜`fix`｜`delete_or_privatize`｜`test_only`（**現状**；目標見 analysis §6 **定案 B**）

呼叫者欄：「生產」= `vllm_omni/` 非 test；「僅 test」= `tests/`；「內部」= 套件內。

---

## 0. 各組件公開 API

只列**對外／公開**表面（`__all__` 或無 `_` 前綴嘅方法／屬性）。私有 `_` 方法見後文逐檔表。

### 0.1 `OmniCoordinatorRuntime`（`runtime.py`）

對外只有呢個 class（`__all__` 有 export）。

| API | 簽名／形狀 | 用途 |
|-----|------------|------|
| `__init__` | `(*, host: str, heartbeat_timeout: float)` | 起 Coord 子進程；暴露地址 |
| `router_address` | `str` | ROUTER 地址 |
| `pub_address` | `str` | PUB 地址 |
| `close` | `() -> None` | 停子進程（idempotent） |

（`run_omni_coordinator_proc`／`_get_*`／`_shutdown_proc` 非對外 API → 見 §5）

### 0.2 `OmniCoordinator`（`omni_coordinator.py`）

| API | 簽名／形狀 | 用途 |
|-----|------------|------|
| `__init__` | `(router_zmq_addr, pub_zmq_addr, heartbeat_timeout=30.0)` | bind ROUTER／PUB，起 recv／periodic thread |
| `get_active_replicas` | `() -> ReplicaList` | 只含 UP 嘅列表 |
| `add_new_replica` | `(event: ReplicaEvent) -> None` | 公開 mutator（生產實際唔經呢度） |
| `update_replica_info` | `(event: ReplicaEvent) -> None` | 同上 |
| `remove_replica` | `(event: ReplicaEvent) -> None` | 同上 |
| `publish_replica_list_update` | `() -> bool` | PUB 當前 active list |
| `close` | `() -> None` | 停 loop、關 socket |
| `wait_for_shutdown` | `() -> None` | 等 stop／join threads |

### 0.3 `OmniCoordClientForStage`（`omni_coord_client_for_stage.py`）

**現状：** 只做 membership（`update`／heartbeat）；**無** bootstrap `register`（仍走 `OmniMasterServer`）。  
**目標（analysis §6 定案 B）：** 擴展為 `register`＋`update`／`heartbeat`（類名可再收斂）；啟動時已知 Coord 地址。

| API | 簽名／形狀 | 用途 |
|-----|------------|------|
| `__init__` | `(coord_zmq_addr, input_address, output_address, stage_id, *, replica_id=0)` | DEALER connect；首發 `update`；起 heartbeat thread |
| `update_info` | `(status=None, queue_length=None) -> None` | 發 `update`（生產少用；多靠 heartbeat hook） |
| `close` | `() -> None` | 發 DOWN、停 heartbeat、關 socket |

工廠：

| API | 簽名／形狀 | 用途 |
|-----|------------|------|
| `create_stage_coord_client` | `(..., *, get_queue_length: Callable[[], int] \| None = None) -> OmniCoordClientForStage` | 建 client 並可掛 `_on_heartbeat` 刷新 queue |

### 0.4 `OmniCoordClientForHub`（`omni_coord_client_for_hub.py`）

**現状：** Head SUB 缓存；Mem／Pool 讀 snapshot。  
**目標（analysis §6）：** **刪除**——`StagePool` 直讀 Coord；唔再經 PUB／SUB／Hub。

| API | 簽名／形狀 | 用途 |
|-----|------------|------|
| `__init__` | `(coord_zmq_addr: str)` | SUB connect；起 recv thread |
| `get_replica_list` | `() -> ReplicaList` | 最新全量 cache（可空列表） |
| `get_replicas_for_stage` | `(stage_id: int) -> ReplicaList` | 按 stage 過濾 |
| `close` | `() -> None` | 停 thread、關 socket |

### 0.5 `LoadBalancer` 家族（`load_balancer.py`）

| API | 簽名／形狀 | 用途 |
|-----|------------|------|
| `LoadBalancingPolicy` | enum：`random`／`round-robin`／`least-queue-length` | CLI／factory 策略名 |
| `Task` | TypedDict：`request_id`／`engine_inputs`／`sampling_params` | `select` 任務上下文 |
| `LoadBalancer.select` | `(task: Task, replicas: list[ReplicaInfo]) -> int` | 抽象：回 `replicas` 下標 |
| `RandomBalancer` | `select(...)` | 均勻隨機 |
| `RoundRobinBalancer` | `__init__(start_index=0)`；`select(...)` | 輪詢 |
| `LeastQueueLengthBalancer` | `select(...)` | 最小 `queue_length`（並列隨機） |

### 0.6 Messages（`messages.py`）

| 型別 | 欄位／成員 | 用途 |
|------|------------|------|
| `ReplicaStatus` | `UP`／`DOWN`／`ERROR` | 副本狀態 |
| `ReplicaEvent` | `input_addr, output_addr, stage_id, event_type, status, queue_length` | Stage→Coord wire |
| `ReplicaInfo` | Event 欄位 + `last_heartbeat, registered_at` | registry／PUB 條目 |
| `ReplicaList` | `replicas: list[ReplicaInfo], timestamp` | Coord→Hub 快照 |

### 0.7 套件根 export（`__init__.py`）

`OmniCoordinator`、`OmniCoordinatorRuntime`、`OmniCoordClientForStage`、`create_stage_coord_client`、`OmniCoordClientForHub`、`ReplicaStatus`、`ReplicaEvent`、`ReplicaInfo`、`ReplicaList`、`Task`、`LoadBalancer`、`LoadBalancingPolicy`、`RandomBalancer`、`RoundRobinBalancer`、`LeastQueueLengthBalancer`。

---

## 1. 廢碼／清理總表（先睇）


| 項目                                     | 位置                                   | 判定                    | 理由                                           |
| -------------------------------------- | ------------------------------------ | --------------------- | -------------------------------------------- |
| `OmniCoordinator.add_new_replica`      | `omni_coordinator.py:87`             | `delete_or_privatize` | 生產／recv 唔呼叫；只經 `_handle_event` → `_*_locked` |
| `OmniCoordinator.update_replica_info`  | `:93`                                | `delete_or_privatize` | 同上                                           |
| `OmniCoordinator.remove_replica`       | `:99`                                | `delete_or_privatize` | 同上                                           |
| `OmniCoordinator.get_active_replicas`  | `:81`                                | `keep`（可改 private）    | 僅 `publish_replica_list_update` 用；無套件外呼叫     |
| `OmniCoordClientForStage.update_info`  | `omni_coord_client_for_stage.py:157` | `test_only`           | 生產路徑唔用；僅 tests                               |
| `ReplicaEvent` 根 export                | `__init__.py`                        | `keep` 或收窄 export     | 套件外幾乎唔 import；wire 內部用                       |
| `Task.engine_inputs`／`sampling_params` | `load_balancer.py:15-24`             | `keep`（標預留）           | 現有 policy 唔讀                                 |
| Diffusion 手寫 `_on_heartbeat`           | `stage_diffusion_proc.py:626-636`    | `fix`（套件外）            | 應改 `create_stage_coord_client`               |


**必須修（行為／可靠性）：**


| 項目                                                      | 位置                                       | 判定    |
| ------------------------------------------------------- | ---------------------------------------- | ----- |
| `_send_event` 對 `zmq.Again` 靜默 drop（含首次 registration）   | `omni_coord_client_for_stage.py:139-141` | `fix` |
| heartbeat 對未知 `input_addr` 忽略                           | `omni_coordinator.py:299-314`            | `fix` |
| `_parse_replica_event`：`queue_length` 未 `int()`         | `omni_coordinator.py:211`                | `fix` |
| Runtime `terminate`／`kill` 唔走 `OmniCoordinator.close()` | `runtime.py:74-81,153-159`               | `fix` |
| close 語意不統一                                             | 見各檔                                      | `fix` |


---



## 2. `__init__.py`

**用途：** 公開 API re-export。


| 符號                               | 生產根 import？                   | 判定                |
| -------------------------------- | ----------------------------- | ----------------- |
| `OmniCoordinatorRuntime`         | 係（`stage_runtime`）            | `keep`            |
| `OmniCoordinator`                | 主要 tests；生產經 `runtime` 子模組    | `keep`            |
| `create_stage_coord_client`      | 係（LLM proc）                   | `keep`            |
| `OmniCoordClientForStage`        | 係（Diffusion 直接用）              | `keep`（目標擴 `register`） |
| `OmniCoordClientForHub`          | Membership 多從子模組 import       | `keep`（目標 **刪除**） |
| `LoadBalancingPolicy`／LB classes | 係                             | `keep`            |
| `ReplicaStatus`／`ReplicaInfo`    | 係                             | `keep`            |
| `ReplicaList`                    | 少；tests／內部                    | `keep`            |
| `ReplicaEvent`                   | 套件外幾乎無                        | `keep` 或唔再 export |
| `Task`                           | 常從 `load_balancer` 子模組 import | `keep`            |


無函式。無廢函式。

---



## 3. `messages.py`

**用途：** wire／domain dataclass；無 I／O。


| 名稱              | 輸入／欄位                                                      | 輸出           | 呼叫者                              | I／O 合理？                  | 判定                |
| --------------- | ---------------------------------------------------------- | ------------ | -------------------------------- | ------------------------ | ----------------- |
| `ReplicaStatus` | enum UP／DOWN／ERROR                                         | —            | 全套件＋pool／membership              | 係                        | `keep`            |
| `ReplicaEvent`  | addrs, stage_id, event_type, status, **queue_length: int** | wire payload | stage client + coordinator parse | 欄位標 int，但 parse 端可塞 None | `keep` + 上游 `fix` |
| `ReplicaInfo`   | 同上 + heartbeats                                            | registry／PUB | coordinator, hub, LB, pool       | 係                        | `keep`            |
| `ReplicaList`   | replicas + timestamp                                       | PUB／cache    | coordinator, hub                 | 係                        | `keep`            |


**可清理：** 無廢 type。注意 `ReplicaEvent.queue_length: int` 同 parse 行為不一致。

---



## 4. `load_balancer.py`


| 名稱                                | 簽名            | 輸入             | 輸出／副作用               | 呼叫者                             | 合理？                               | 判定     |
| --------------------------------- | ------------- | -------------- | -------------------- | ------------------------------- | --------------------------------- | ------ |
| `Task`                            | TypedDict     | request_id 等   | —                    | `stage_pool.pick` 只填 request_id | 額外欄位預留                            | `keep` |
| `LoadBalancingPolicy`             | Enum          | —              | —                    | serve CLI、stage_runtime factory | 係                                 | `keep` |
| `LoadBalancer.select`             | abstract      | task, replicas | index                | 子類                              | 係                                 | `keep` |
| `RandomBalancer.select`           |               | replicas 非空    | random index         | 生產                              | 空 list raise OK                   | `keep` |
| `RoundRobinBalancer.__init__`     | start_index=0 | —              | counter+Lock         | factory／tests                   | 係                                 | `keep` |
| `RoundRobinBalancer.select`       |               | replicas       | RR index             | 生產                              | 順序敏感（doc 已寫）                      | `keep` |
| `LeastQueueLengthBalancer.select` |               | replicas       | min queue；tie random | 生產                              | 負 queue raise OK；依賴 heartbeat 新鮮度 | `keep` |


**可清理：** 無廢 function。`Task` 多餘欄位可註明 unused-by-current-policies。

**結構備註（P2）：** 本檔**無 ZMQ、唔識 Coord 進程**；生產由 `StagePool.pick` 擁有。放喺 `omni_coordinator/` 只係歷史打包＋共用 `ReplicaInfo`。功能 `keep`；**套件歸屬宜另議遷出**（見 analysis §3 P2）。

---



## 5. `runtime.py`


| 名稱                                | 簽名                               | 輸入      | 輸出／副作用                                                | 呼叫者                 | 合理？             | 判定                     |
| --------------------------------- | -------------------------------- | ------- | ----------------------------------------------------- | ------------------- | --------------- | ---------------------- |
| `run_omni_coordinator_proc`       | router, pub, timeout, ready_pipe | pipe    | 建 OC → ready → `wait_for_shutdown`；**永遠唔 call close** | Process target      | 子進程只靠殺          | `fix`（應可 signal→close） |
| `_get_coordinator_mp_context`     | ()                               | —       | fork 優先 else spawn                                    | Runtime.**init**    | TODO spawn-safe | `keep` + 跟進 TODO       |
| `_shutdown_proc`                  | proc                             | Process | terminate→join→kill                                   | close／finalizer     | 粗暴但明確           | `fix`（配優雅關閉）           |
| `OmniCoordinatorRuntime.__init__` | host, heartbeat_timeout          | 校驗      | 配 port、spawn daemon、等 ready 30s                       | `stage_runtime:929` | 單實例＝SPOF 根源     | `keep`（行為需文件化）         |
| `OmniCoordinatorRuntime.close`    | ()                               | —       | **idempotent**；kill 子進程                               | `stage_runtime:848` | 同其他 close 語意唔齊  | `fix`                  |


**可刪：** 無。  
**必須修：** 關閉路徑應觸發 `OmniCoordinator.close()`；daemon／單實例寫入系統風險文件（已完成）。

---



## 6. `omni_coordinator.py`



### 5.1 Public


| 名稱                            | 輸入                       | 輸出／副作用                             | 呼叫者                                 | 合理？                  | 判定                             |
| ----------------------------- | ------------------------ | ---------------------------------- | ----------------------------------- | -------------------- | ------------------------------ |
| `__init__`                    | router／pub addr, timeout | bind；起 recv＋periodic threads       | runtime, tests                      | 建構即開線程，難測但現況可用       | `keep`                         |
| `get_active_replicas`         | —                        | ReplicaList(UP only)               | **僅** `publish_replica_list_update` | OK                   | `keep`→可 `_` 私有                |
| `add_new_replica`             | ReplicaEvent             | 鎖＋schedule broadcast               | **無**（生產／recv 都唔用）                  | 同 `_handle_event` 重複 | `delete_or_privatize`          |
| `update_replica_info`         | ReplicaEvent             | 同上                                 | **無**                               | 重複                   | `delete_or_privatize`          |
| `remove_replica`              | ReplicaEvent             | mark DOWN＋broadcast                | **無**                               | 重複                   | `delete_or_privatize`          |
| `publish_replica_list_update` | —                        | PUB JSON NOBLOCK；bool              | `_periodic_loop`                    | drop 靜默＝best-effort  | `keep`；文件化 P1                  |
| `close`                       | —                        | join threads；關 socket；**二次 raise** | 幾乎僅 tests                           | 生產 Runtime 唔叫        | `fix`（idempotent + Runtime 接通） |
| `wait_for_shutdown`           | —                        | block `_stop_event`                | runtime child                       | OK                   | `keep`                         |




### 5.2 Private


| 名稱                            | 輸入           | 輸出／副作用                            | 呼叫者           | 合理？                                                    | 判定                    |
| ----------------------------- | ------------ | --------------------------------- | ------------- | ------------------------------------------------------ | --------------------- |
| `_schedule_broadcast`         | —            | set pending flag                  | 多處            | OK coalesce                                            | `keep`                |
| `_mark_replica_error_locked`  | info         | status=ERROR                      | timeout check | OK                                                     | `keep`                |
| `_check_heartbeat_timeouts`   | —            | UP→ERROR；DOWN／ERROR 600s GC       | periodic      | OK                                                     | `keep`                |
| `_parse_replica_event`        | dict         | ReplicaEvent｜None                 | recv          | `queue_length=data.get(...)` 可 None，同 dataclass int 衝突 | `fix`                 |
| `_recv_loop`                  | —            | ROUTER recv→handle                | daemon thread | Again continue OK                                      | `keep`                |
| `_periodic_loop`              | —            | timeout＋keepalive PUB             | daemon thread | keepalive 設計合理                                         | `keep`                |
| `_handle_event`               | ReplicaEvent | 分派 heartbeat／register／update／down | recv          | **heartbeat 未知 addr 忽略**（L303-314）                     | `fix`                 |
| `_add_new_replica_locked`     | event        | 寫 `_replicas`                     | handle        | 校驗 input_addr／stage_id                                 | `keep`                |
| `_update_replica_info_locked` | event        | 改 status／queue                    | handle        | OK                                                     | `keep`                |
| `_remove_replica_locked`      | event        | status=DOWN（唔即刪）                  | handle        | 靠 GC；同「remove」字面略偏                                     | `keep`（可改名 mark_down） |


**可刪／收窄：** 三個 public mutator。  
**必須修：** parse queue_length；heartbeat upsert 或明確 log。

---



## 7. `omni_coord_client_for_stage.py`


| 名稱                          | 輸入                                | 輸出／副作用                                           | 呼叫者                              | 合理？                         | 判定                |
| --------------------------- | --------------------------------- | ------------------------------------------------ | -------------------------------- | --------------------------- | ----------------- |
| `__init__`                  | coord addr, in／out addr, stage_id | connect；`_send_event("update")`；起 heartbeat      | factory／Diffusion／tests          | **首次 update 可被 Again drop** | `fix`             |
| `_reconnect`                | max_retries=3, interval=5         | 重建 DEALER                                        | `_send_event`                    | 有限次；長期 down 後放棄             | `keep`（可調）        |
| `_send_event`               | event_type                        | NOBLOCK send；Again→**drop return**；其他錯→reconnect | init／update／hb／close             | **P1 不可靠**                  | `fix`             |
| `update_info`               | status? queue_length?             | 改本地＋send update                                  | **僅 tests**                      | API 合理但生產唔用                 | `test_only`       |
| `_heartbeat_loop`           | —                                 | 每 5s hook＋heartbeat                              | daemon                           | hook 例外被吞（應有）               | `keep`            |
| `close`                     | —                                 | stop；DOWN update；關 socket；**二次 raise**           | LLM／Diff finally                 | DOWN 亦可能 Again drop         | `fix`（idempotent） |
| `create_stage_coord_client` | + optional queue_length_getter    | 建 client＋掛 `_on_heartbeat`                       | LLM `stage_engine_core_proc:162` | 正確公開入口                      | `keep`            |


**可清理：** 鼓勵刪／標記 `update_info` 為 test helper；Diffusion 改用 factory。  
**必須修：** registration／critical update 唔可 Again 靜默成功返回。

---



## 8. `omni_coord_client_for_hub.py`


| 名稱                       | 輸入       | 輸出／副作用                              | 呼叫者                  | 合理？                     | 判定                |
| ------------------------ | -------- | ----------------------------------- | -------------------- | ----------------------- | ----------------- |
| `__init__`               | pub addr | 起 thread；等 init 5s；失敗 raise         | MembershipController | fail-fast OK            | `keep`            |
| `_decode_replica_list`   | dict     | ReplicaList                         | recv                 | `int(queue_length)` 嚴格  | `keep`            |
| `_recv_loop`             | —        | SUB connect／recv／reconnect；更新 cache | daemon               | Coord 長期 down 時 1s 空轉   | `keep`            |
| `get_replica_list`       | —        | 最新 cache 或空 list                    | membership watcher   | OK                      | `keep`            |
| `get_replicas_for_stage` | stage_id | filter                              | StagePool            | OK                      | `keep`            |
| `close`                  | —        | stop＋join 1s；**二次 raise**           | Membership.shutdown  | join timeout 後仍標 closed | `fix`（idempotent） |


**可刪：** 無廢 function。  
**可收窄：** 無同步 snapshot RPC（系統層 P1／設計項，唔係廢碼）。

---



## 9. 跨檔 I／O 合理性摘要


| 路徑                         | 問題                               | 嚴重度          |
| -------------------------- | -------------------------------- | ------------ |
| Stage → Coord registration | `Again` drop 當成功                 | P1           |
| Stage heartbeat → Coord    | 未知 addr 忽略                       | P1           |
| Coord → Hub PUB            | NOBLOCK drop；靠 keepalive 補       | P1／可接受若有 SLA |
| Parse queue_length         | 可 None 入 int 欄位                  | P2           |
| Runtime shutdown           | kill 唔 close                     | P1           |
| Hub decode                 | 嚴格 int — 同 coordinator parse 不對稱 | P2           |


---



## 10. 建議 PR 切分（對齊 analysis §6）

系統級切分以 analysis 為準（唔用本節舊 A–F 字母，避免同「定案 B」撞名）：

| PR | 內容 | 對應 |
|----|------|------|
| **PR1** | SPOF／註冊測試；R1–R5；死 API；統一 LLM／Diff 路徑；LB 歸屬（仍只內建三策略） | P0＋P1 可靠性＋P2 小清理 |
| **PR2** | Master→Coord；**ClientForStage 兼 `register`**；**刪 Hub／Mem**；Pool 直讀；dotted-path 自訂 LB | P1 重構主線＋自訂 LB |

本 audit 逐項修復（Again／upsert／close／int／dead API）優先落入 **PR1**。

---



## 11. 覆蓋檢查清單

- [x] `__init__.py`
- [x] `messages.py`
- [x] `load_balancer.py`（全部 select／**init**）
- [x] `runtime.py`（全部 top-level + Runtime）
- [x] `omni_coordinator.py`（全部 public／private）
- [x] `omni_coord_client_for_stage.py`
- [x] `omni_coord_client_for_hub.py`
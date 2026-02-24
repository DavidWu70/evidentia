# Evidentia MVP 状态与任务台账

更新时间: 2026-02-23
对应任务: `evidentia/Codex-Ready_Task_v4 pre.md`

## 1) 当前状态快照

已完成:
- 新骨架目录可用: `evidentia/backend`、`evidentia/frontend`、`evidentia/docs`
- 后端 MVP 接口已实现:
  - `GET /note/templates`
  - `POST /transcribe_structured`
  - `POST /transcript/incremental`
  - `POST /events/extract`
  - `POST /state/snapshot`
  - `POST /note/draft`
- Note 模板已从代码内置改为文件化配置:
  - `backend/note/templates/<department>/*.txt`（兼容 `*.md`）
  - 前端模板列表改为从后端 `GET /note/templates` 动态加载
  - 前端已提供可视化模板编辑器（表单编辑，后台保存为文本模板）
- 前端三栏评审页已可跑通:
  - transcript
  - evidence timeline
  - risk + editable draft note
- 前端主链路已切到 `POST /transcribe_structured`，支持两种输入模式:
  - `transcript`（准流式）
  - `audio_path`（真实 ASR 分块流式入口）
- 前端已增加 `Pipeline Debug` 卡片，可直接验证:
  - `source/asr/diarization` 状态
  - 事件抽取实际引擎（`medgemma` 或 `rule_fallback`）
- Open Questions 已升级为 Hybrid 结构:
  - `mandatory_safety_questions`（规则主系统）
  - `contextual_followups`（可选 AI 增强）
  - `rationale`（可解释性文本）
  - 保持 `open_questions` 合并字段向后兼容
- 风险提示与免责声明已展示
- 后端测试通过: `backend/tests`（当前 68 passed）
- `audio_path` diarization 第一阶段（完整链路）已接入:
  - backend 新增 diarization adapter（VAD/SAD + embedding + clustering + turn segmentation）
  - `/transcribe_structured` `utterances/new_utterances` 已输出 `speaker_id/speaker_role/diar_confidence`
  - ASR + diarization 对齐写入 utterance，不再仅依赖 `other` 回退标签
  - diarization 失败时保留 fallback（优先 ASR turns，再回退 inferred turns）
- 角色映射第一阶段（`speaker_id -> patient/clinician/other`）已接入:
  - `audio_path` 与 `ws/audio/live` 均启用会话级稳定映射（含状态保持与 `reset` 清理）
  - utterance 新增 `role_confidence`，低置信度保持 `other`（不强制误标）
- MedGemma 抽取已做轻量稳定性增强（默认逐句模式）:
  - 默认 `EVIDENTIA_MEDGEMMA_PER_UTTERANCE=1`，每条 utterance 单独推理并使用紧凑提示词
  - 增加 `type-label` 一致性校验（非法组合如 `symptom + passive_suicidal_ideation` 会被丢弃）
  - `event_harmonization` 当前为非破坏性（不再抑制 `sleep_disturbance`）

当前限制:
- 已接入真实音频 ASR 适配层（repo chunk pipeline），但运行依赖本机 provider 配置
- 事件抽取支持 MedGemma 与规则 fallback，但 MedGemma 运行依赖本机 GGUF + llama_cpp
- 前端已迁移为最小 React 版本（ESM CDN 方式），仍可继续做工程化构建
- MedGemma JSON 产出稳定性仍需继续调优（当前已改为增量抽取 + 局部回看，但模型输出质量仍依赖提示词与模型行为）
- `auto` 模式已切为“规则实时 + MedGemma 异步回填（按窗口间隔）”，仍需做长会话稳定性基线验证
- Open Questions AI 增强依赖本地模型（llama_cpp + GGUF）；模型不可用时自动降级为规则-only，不阻断主流程
- 当前“实时”主链路中，`transcript` tick 与 `audio_path` window 仍是轮询/窗口化准实时；麦克风链路已改为 WS 事件驱动更新（`ack_chunk` 携带增量 utterances）
- 麦克风链路已补基础“增量转录拉取”通道：`GET /audio/live/transcript/{session_id}`（作为 fallback/debug）；前端 `Start Mic` 默认走 WS `ack_chunk` 事件驱动消费增量 utterances；默认已接入 repo chunk ASR 适配（可被 `live_audio_chunk_asr_callable` 覆盖）
- 当前会话状态主要在内存中，尚未落库；系统重启后无法按 `session_id` 做历史回放与审计查询
- 已有 ws 麦克风增量 diarization；尚未完成跨窗口在线重聚类与长会话角色精度基线
- `speaker_role` 目前为启发式映射（`patient/clinician/other`），临床场景仍需继续校准

## 1.1) 已实施设计（2026-02-20）

目标:
- 将 `audio_path` 的事件抽取改为“增量优先 + 临近窗口回看 reconcile”，避免全量重算导致耗时和 JSON 失稳。

参数约定（前端 UI）:
- `reconcile_lookback_windows`（整数）
  - `0`: 关闭 reconcile（仅增量窗口抽取）
  - `N>0`: 每个窗口在“当前窗口 + 前 N 个窗口”范围做局部 reconcile

设计要点:
- 后端 `transcribe_structured` 返回 `new_utterances`（仅本次新增）
- 前端事件抽取只提交 `new_utterances`（非全量 utterances）
- 前端本地维护事件去重 store（按 `segment_id+type+label+polarity` upsert）
- 窗口回看只针对最近 N 个窗口，不做整段全量回看
- 防止大块拼接：禁止跨窗口 utterance merge，必要时按长度/时长强制切分

## 2) 与 v4 任务文档差距

高优先级差距:
- `POST /transcribe_structured` 已在新骨架实现并接入 ASR adapter，仍需稳定性验证
- 真实 ASR 音频转写链路已接入，但仍需 sample audio 端到端稳定性验证
- MedGemma 推理抽取链路已接入，仍需真实模型质量评估与提示词调参

中优先级差距:
- 测试覆盖未达到任务要求:
  - e2e smoke 已有脚本，仍可补更多真实音频样本
- Demo hardening 已有基础能力（fallback checks + UI debug），仍可继续补:
  - 固定 demo 音频清单与质量基线
  - 真实长音频场景下的分块参数调优（window/chunk/overlap）

## 3) 后续任务登记（按执行顺序）

### P0: 打通真实输入与模型链路
- [x] 任务 P0-1: 在 `evidentia/backend` 增加 `POST /transcribe_structured`
- [x] 任务 P0-2: 接入 ASR 适配层（音频 chunk -> ASR segments）
- [x] 任务 P0-3: 增加 MedGemma 抽取适配层（utterances -> events）
- [x] 任务 P0-4: 保留规则 fallback（模型不可用时回退）

完成定义:
- 可用 sample audio 触发端到端流程
- 能稳定产出 evidence-backed events
- 模型失败时系统可降级但不中断

### P1: 前端到评审可演示质量
- [x] 任务 P1-1: 将当前 `frontend` 迁移为最小 React 实现
- [x] 任务 P1-2: 增加证据点击联动（timeline -> transcript 定位）
- [x] 任务 P1-3: 增加错误态与空数据态提示
- [x] 任务 P1-4: 保留 note 可编辑 + export

完成定义:
- 三栏 UI 全流程可演示
- 关键事件可点击并定位证据
- 错误态不会导致页面“假成功”

### P2: 测试与发布前固化
- [x] 任务 P2-1: 补 unit tests（events/risk/note）
- [x] 任务 P2-2: 补 API tests（含异常路径）
- [x] 任务 P2-3: 增加 e2e smoke（sample transcript/audio）
- [x] 任务 P2-4: 更新 runbook 与 demo script

完成定义:
- 核心路径与关键异常路径可重复通过
- 按 runbook 可在新环境复现演示

### P3: 评审演示硬化
- [x] 任务 P3-1: 前端切换到 `/transcribe_structured` 主链路
- [x] 任务 P3-2: 增加 `audio_path` 模式用于真实 ASR 演示
- [x] 任务 P3-3: 增加 UI 级 pipeline debug，可确认 ASR/MedGemma 实际使用
- [x] 任务 P3-4: 增加固定备援检查脚本（ASR 失败 / MedGemma 失败）
- [x] 任务 P3-5: `audio_path` 改为分块流式 UI（非 one-shot）

完成定义:
- 评审现场可以直接从 UI 判定“是否真正使用 ASR 与 MedGemma”
- 至少有一套失败场景可快速切换演示 fallback 逻辑

### P4: MedGemma 分块抽取与稳定性硬化（进行中）
- [x] 任务 P4-1: `transcribe_structured` 增加 `new_utterances` 响应字段
- [x] 任务 P4-2: 前端改为“仅对 `new_utterances` 调用 `/events/extract`”
- [x] 任务 P4-3: 增加 `reconcile_lookback_windows` 参数（`0=关闭`，`N>0=前N窗口回看`）
- [x] 任务 P4-4: 增加跨窗口 merge 限制，避免单条 utterance 无限膨胀
- [x] 任务 P4-5: `auto` 改为“rule 实时 + medgemma 异步回填”，减少主链路阻塞
- [ ] 任务 P4-6: 补充长会话 e2e（验证性能与 JSON 产出稳定性）

完成定义:
- 长会话不再执行全量事件重算
- MedGemma 提示词长度与单窗口规模同阶
- `reconcile_lookback_windows=0` 时可稳定纯增量运行

### P5: 实时转录主链路（下一优先级）
- [x] 任务 P5-1: 增加麦克风实时输入链路（浏览器采集 -> 后端流式接收）
- [ ] 任务 P5-2: 增量转录低延迟路径（目标首字延迟与稳定更新频率可观测）
  - 已完成基础版：`/audio/live/transcript` 增量拉取 + Pipeline Debug 观测
  - 已完成增强版：麦克风链路改为滑动窗口增量 ASR（默认 20s 窗口 / 4s 推送步长 / 1s 采样）
  - 已完成增强版：`ws/audio/live` 增量 diarization（默认本地 adapter，可注入 callable；失败自动回退 `other`）
  - 已完成增强版：`ws/audio/live` 支持句级角色拆分（命中单 speaker_id + 混合句式时触发），并在 `ack_chunk`/`audio/live/transcript` 暴露 `sentence_role_split` debug
  - 已完成可配置化：`Live Mic Slice (ms)`、`Live ASR Window (s)`、`Live Push Step (s)` 已接入 Config 页面
  - 未完成闭环：首字延迟 SLA 基线与压测（长会话）
- [ ] 任务 P5-3: 将 `Live Transcript` 与 `Evidence Timeline` 切为事件驱动更新（减少轮询抖动）
  - 已完成：麦克风链路（`Start Mic`）使用 WS `ack_chunk` 事件驱动刷新
  - 待统一：`audio_path` window 模式仍为轮询 tick
- [x] 任务 P5-4: 增加实时链路断连恢复与降级策略（自动重连/降级实时缓存模式）
  - 已完成：前端本地未发送 chunk 缓存 + WS 自动重连补发，后端按 `seq` 去重
  - 已完成：重连超阈值后，前端自动将缓存音频拼接为文件并切换到 `audio_path` fallback
- [x] 任务 P5-5: 增加跨页面刷新/崩溃后的缓存恢复（持久化队列）与更细粒度降级策略
  - 已完成：未发送 mic chunk 从内存队列升级为 IndexedDB 持久化队列（支持刷新/崩溃后恢复补发）
  - 已完成：`Stop Mic` 改为优雅停止（先停止采集，等待队列排空后再关闭 WS；超时进入 `stopped_with_pending_queue`）

完成定义:
- UI 在会话进行中持续看到新转录与新证据，无需手动刷新
- 中断/重连后会话不中断且不会产生明显重复片段
- Pipeline Debug 能明确显示实时链路状态（connected/reconnecting/degraded）

### P5.5+: 完整 diarization 第一阶段（audio_path）
- [x] 任务 P5.5+-1: 新增 backend diarization adapter（VAD/SAD + embedding + clustering + turn segmentation）
- [x] 任务 P5.5+-2: `/transcribe_structured`（`audio_path`）输出 `speaker_id/speaker_role/diar_confidence`
- [x] 任务 P5.5+-3: ASR+diarization 对齐后写入 utterances（不再全部 `other`）
- [x] 任务 P5.5+-4: 前端 Transcript 保持 `speaker` 展示，并支持显示 `speaker_id/role`
- [ ] 任务 P5.5+-5: 长会话与真实音频角色映射精度基线（门诊/心理访谈场景）
- [x] 任务 P5.5+-6: `speaker_id -> patient/clinician/other` 会话级稳定映射（含 fallback 与 reset）

完成定义:
- `audio_path` 模式返回的 utterance 可同时携带 `speaker/speaker_id/speaker_role/diar_confidence`
- `audio_path` 与 `ws/audio/live` 的 utterance 可携带 `role_confidence`，并基于会话级证据稳定输出角色
- diarization 失败不阻断链路，仍可返回可用 transcript（`other` fallback）
- 前端无需切换模式即可在 transcript 卡片查看 speaker id/role

### P6: Postgres 持久化 + 回放/审计查询
- [ ] 任务 P6-1: 设计并落地会话数据模型（患者、科室配置、模板选择、转录段、事件、快照、草稿）
- [ ] 任务 P6-2: 增加写入路径（会话进行中增量落库，失败可重试）
- [ ] 任务 P6-3: 增加查询接口（按 `session_id` 重建 Transcript/Timeline/Note）
- [ ] 任务 P6-4: 增加审计视图字段（模型版本、模板版本、关键参数、时间戳）

完成定义:
- 服务重启后仍可按 `session_id` 完整回放会话结果
- 可按患者/时间范围检索历史会话
- 审计接口可解释“当时为什么得到该风险与草稿”

### P7: Notes Tab UX Upgrade + Graceful Stop Generation
- [ ] Task P7-1: Refactor Notes tab to two-column layout (template tree on left, note content on right)
- [ ] Task P7-2: Render templates as department-based tree with multi-select checkboxes
- [ ] Task P7-3: Add per-template generation status badges (Queued/Generating/Generated/Failed/Stopped)
- [ ] Task P7-4: Add generated-state indicator (green check) and click-to-open generated note in right pane
- [ ] Task P7-5: Replace one-shot generate with job-style APIs (`start/status/stop`) and frontend polling
- [ ] Task P7-6: Implement graceful stop semantics (`Stopping...` in UI, backend stops after current template)

Definition of Done:
- Notes tab supports multi-template generation with visible per-template progress
- `Generate Notes` switches to `Generating...` and becomes unavailable during job execution
- `Stop` is visible only while generating/stopping and transitions UI to `Stopping...`
- Backend stop is graceful (no hard kill): current template can finish, remaining templates are skipped

## 4) 范围锁定（防止偏航）

本阶段只做:
- 3 类事件: `risk_cue`、`symptom`、`duration_onset`
- 模板化 note 生成（当前支持 `psych` 与 `internal_med` 的文件模板）
- 评审级本地演示，不做生产级部署

本阶段不做:
- 自动诊断结论
- 自动治疗建议
- 多租户权限与合规全套

## 5) 更新规则

每次完成一个任务时，必须同步更新:
- 本文件“后续任务登记”中的勾选状态
- “当前状态快照”中的已完成与当前限制
- 更新时间

建议提交节奏:
- 每完成一个 P0/P1 任务就进行一次提交并更新本台账

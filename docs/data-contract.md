# 带相关发布噪声的课程参与统计及变化可信度：业务资料契约

releases 给出不可变 release_id、replaces 和 measurements；每项是 cells 中潜在人数组合的线性和，coefficients 的键为 cell_id。mechanisms 的 loading 指向共同噪声变量，noise_variance 定义其已知方差，独立噪声项也以独立变量表示；同键跨发布共享同一扰动。constraints 是真实分类关系而非修数目标。cells 中未覆盖人群不可由发布噪声恢复。

所有示例记录由本地确定性生成，不含真实居民、客户或学校个人资料。`revision` 指向整包不可变资料，`synthetic=true` 只说明资料性质，不声明统计结论。

## 当前产品边界

现有分析 `release_inventory` 已可通过资料目录服务、分析服务和结果回读完整运行；它只给已观察记录的业务核对，不实现 `latent_participation`。新分析需要与当前核对功能并存，统计方法、模型分解和诊断组织由实现者根据证据选择。

新分析的最小产品交付是可保存的结构化分析结果，包含目标人群/估计对象、所用资料及排除原因、估计或可识别范围、适用条件、诊断和业务解释；无法估计时给原因而不是伪造数值。`options` 应允许调用者保存并重放题面涉及的假设或情景，具体字段可随实现补充文档。默认调用用资料能够支持的基本情景，不能把未声明的强假设隐藏进默认值。数值容差、随机种子和蒙特卡洛精度要可复查。

## 已有业务投影视图

`app/releases.py` 的 `release_profile` 为当前核对分析生成 `source_profile`，保留此项目的分析单位、重复观测和资料组织。这些已观察资料的投影不是新分析结果；可复用其数据关系，但不能把描述统计当成推断。修改现有资料处理时要保留 `tests/test_context.py` 的业务不变量。

## `latent_participation`：参与规模与变化分析

`latent_participation` 与 `release_inventory` 并存在 `ANALYSES` 中。模型取资料声明的线性可加形式：

```
y = A x + L e,     e ~ N(0, diag(noise_variance)),   S = L diag(v) L^T
```

`x` 是各期互斥分类组合的潜在人数，`A` 由 `coefficients` 构成，`L` 由 `loading` 构成。估计用带约束的广义最小二乘，完整使用 `S`，因此：

- 同一次发布的总计与细分项若共用噪声变量，不被当作独立观测；
- 更正版若与旧版共享 `shared`/`new` 等变量，不被当作独立新增样本，等权平均不会带来 `1/sqrt(n)` 的方差下降；
- `S` 的零方差方向是 `x` 上的精确线性关系，按精确关系处理，而不是丢弃；
- 设计矩阵、取值与载荷完全相同的两行只算一次观测。真正重复发布的资料在结果上完全等价，重复行在 `noise.redundant_republished_measurements` 中列明。

硬约束来自资料本身，不以放宽约束换拟合：`x >= 0`、`fixed_total` 各期总人数、`linear`/`relation` 声明的跨表线性关系。未识别的 `kind` 会进入 `data.constraints_not_applied`，不会被静默忽略。

### `options`

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `confidence_level` | `0.90` | profile 区间水平 |
| `consistency_level` | `0.05` | 判定“统计张力”的显著性水平 |
| `release_scope` | `"current"` | `current` 只用未被 `replaces` 命名的发布；`all` 全用；`ids` 用 `release_ids` |
| `release_ids` | `null` | 仅 `release_scope="ids"` 时允许 |
| `include_superseded` | `false` | `current` 下是否也纳入被替代的旧发布 |
| `subset_limit` | `12` | 做发布组合一致性枚举的最大发布数，超过则不做枚举 |

未知字段一律拒绝（422），避免把未声明的强假设悄悄塞进默认值。所有选项会原样写入 envelope 的 `options`，可重放。

### 结果结构

- `estimand` / `scenario`：估计对象、单位、所用的情景与选项。
- `data`：所用与排除的发布及原因、被替代发布、`raw_release_values`（原始发布值，含负人数，永久可查）、噪声方差、已应用与未应用的约束、疑似残缺的 cell 组合。
- `fit`：可行性、是否估计成功、目标函数值、自由度、p 值、约束违背量、独立观测数、残差与说明。
- `periods`：每期每个 cell 与聚合量（各项目人数、参与人数、总人数）的估计、区间、结构性范围、可识别性。不可识别的量不给点估计也不给噪声区间。
- `changes`：相邻期之间的变化，含区间与 `direction_resolved`；被声明约束钉死的量会标注 `pinned_by`。
- `noise`：噪声变量清单、冗余重复行、等权平均口径的低估倍数 `understatement_factor`。
- `coverage`：未被交换资料覆盖的人群。它们不会被填成 0，也不会进入区间；若由声明约束唯一确定则会标注。
- `contradictions`：`structurally_feasible`、`jointly_consumable`、`infeasibility_witness`、`maximal_consistent_release_sets`、`conflicting_groups`。相互矛盾的发布不会被修成一张表。
- `identities`：逐期总人数、参与人数与两者的关系，均来自同一约束系统。
- `conditions`、`interpretation`：适用条件与面向运营的中文解释。

### 不确定性的含义

区间是**发布噪声**下的 profile 似然区间，不是对居民或学员总体的抽样推断。默认无随机数、无蒙特卡洛，因此没有蒙特卡洛误差；`method` 中记录 `random_seed: null`、`monte_carlo: false` 与数值容差。点估计与区间对发布/测量行的排列不变（差异仅在 SLSQP 容差 `1e-7` 量级）。

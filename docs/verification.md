# 可观察的验证边界

当前 `tests/test_product.py` 检查真实 HTTP 创建与回读、结果内容身份、未知分析拒绝及资料版本；`tests/test_domain.py` 检查此产品特有的数据语义。它们是已有功能回归测试，不是假装新功能已经交付的通过凭证。

新增 `latent_participation` 后请补充领域测试。需要覆盖：相关噪声机制与重复发布依赖; 潜在人数组合约束与不确定性; 不兼容发布证据诊断; 真实变化与发布误差的区分。可使用更小的可人工核对合成资料、观测顺序置换与无支持/退化资料验证，涉及随机推断时说明精度。不得以当前示例产生某种显著性方向作为验收；资料不足、零差异或相互冲突的结果都可以是正确输出。

正式展示使用 `python -m app.demo --analysis latent_participation`，从空临时结果目录启动两个真实服务、运行分析并完整回读保存结果。该名字是为同一产品登记的新分析键，不预设具体数值或结论。初始版本尚未登记此键，此命令会明确拒绝；当前运行方式见 README。开发后正式运行仍应保持 90 秒以内，不下载资料、模型或依赖。展示输出应包含统计证据而非只有成功状态。

开发后另外运行 `python -m unittest tests.feature_acceptance -v` 检查新分析接入与来源不变性。这组测试在初始版本因新分析未登记而失败，不属于当前产品基线通过项，也不替代统计正确性验证。三份领域边界资料见 docs/analytical-cases.md，其当前资料读取可由 tests/test_cases.py 验证。

## `latent_participation` 的领域测试

`tests/test_latent.py` 覆盖并对应到 `docs/verification.md` 要求的四类：

- 相关噪声机制与重复发布依赖：`test_duplicate_republication_adds_no_information`（完全相同取值的重复发布不改变估计、p 值和独立观测数，并报告冗余行）、`test_correction_sharing_noise_is_not_independent`（更正版与旧版共享噪声，`understatement_factor > 1`，说明等权平均会低报不确定性）。
- 潜在人数组合约束与不确定性：`test_counts_respect_non_negativity_and_period_totals`、`test_intersection_and_margins_are_mutually_consistent`、`test_noiseless_case_is_exactly_recovered`（boundary-1 可人工核对）、`test_declared_linear_relation_is_enforced`。
- 不兼容发布证据诊断：`test_conflicting_noiseless_releases_are_reported_not_repaired`（boundary-3 给出 witness，不产出任何点估计）、`test_joint_tension_lists_maximal_consistent_releases`。
- 真实变化与发布误差的区分：`test_changes_report_uncertainty_direction`（方向判定必须与区间一致，而不是只看点估计符号）、`test_declared_total_is_not_given_a_noise_interval`。
- 资料覆盖与噪声的分界：`test_uncovered_population_is_not_filled_in`（未交换人群不给点估计与区间）。
- 可复查性：`test_analysis_is_deterministic`、`test_row_order_does_not_change_results`（发布/测量/cell 行序置换，估计与区间在 `1e-4` 内一致）。

这些测试不预设效应方向或显著性；`data/current.json` 在当前默认口径下各发布可以联合解释，`release_scope="all"` 时存在统计张力，两种结果都被视为正确输出。

# 数据与现有口径

kind 为 exposure/click/purchase，available 是记录时刻的可售标记。

`data/records.json` 是版本为 1 的演练数据，所有人员、机构、商品和交易均为本地合成记录。`id` 是导入记录键，不自动代表企业、访问或其他统计单位；记录键不得重复。数值字段必须有限，整数字段不得超过安全整数范围。空字符串可用于字段约定允许的未公开标签，不用零冒充缺测。

| 字段 | 类型 |
| --- | --- |
| `id` | `string` |
| `user` | `string` |
| `item` | `string` |
| `time` | `integer` |
| `kind` | `string` |
| `strength` | `number` |
| `available` | `boolean` |

现有 `summary` 按 `kind` 清点记录，`sum` 为 `strength` 的逐行核对值，不代表统计推断或跨期业务指标。分组按键排序，行按记录键稳定处理。`data/expected-summary.json` 是独立生成的现有核算对照；`data/policy.json` 保存业务配置的演练值，当前清点命令不消费分析配置。

`inspect` 返回输入行数、字段列表与 SHA-256 指纹；指纹不依赖字段或记录排列。数据版本不支持时应直接拒绝，不做静默迁移。错误发生时命令返回非零退出码。

## 推荐器口径

`train` 消费 `data/policy.json` 的分析配置（`summary`/`inspect` 仍不消费）。训练窗口为 `time < trainBefore`，验证窗口为 `[trainBefore, validationBefore)`；窗口之后的记录两侧都不进入。模型文件保存用户/商品映射、因子、训练时间界限、热门兜底表与可售时间线，`recommend` 加载后对相同请求结果一致。

### 配置字段（`data/policy.json`）

| 字段 | 含义 |
| --- | --- |
| `trainBefore` / `validationBefore` | 训练/验证时间界限 |
| `rank` / `topK` / `seed` | 隐因子维度、推荐列表长度、初始化种子 |
| `iterations` / `regularization` | ALS 迭代轮数与 L2 正则 |
| `alpha` | 置信权重系数：`c = 1 + alpha × 加权互动强度`，重复点击与高强度互动由此放大置信 |
| `kindWeights` | 各记录类型的强度权重；`exposure` 默认 0，可调大把曝光当弱正例 |
| `exposureConfidence` / `exposurePreference` | 已曝光未互动对的软置信/软偏好 |
| `unobservedConfidence` | 未曝光对的均匀先验置信：不当作负例、不随热度加权，仅避免未曝光商品得分无约束浮高，可置 0 关闭 |
| `longTailShare` | 头部商品累计覆盖的互动强度比例，其余为长尾 |

### 评估口径

- 相关商品：验证窗口内强度大于 0 的互动（点击/购买）商品；曝光不计入相关。
- 候选空间：请求时刻可售（按该时刻之前最近的可售标记判定）且模型可评分的商品，逐用户记录在 `metrics.json` 的 `users[].candidates`；不可售商品列入 `excluded` 并说明原因。可售标记是目录状态而非用户行为，不读取请求时刻之后的记录。
- 分数来源：活跃用户（训练期互动强度大于 0）为 `mf` 隐因子点积；其余为 `popularity-fallback` 训练期热门兜底并附原因；基线为 `popularity-baseline`，与模型共用候选约束。
- 用户切片：`active` 与 `cold-start` 互斥；`new`（训练窗口无记录）、`singleExposure`（仅一次曝光）、`zeroInteraction`（有互动记录但强度全为 0）为可叠加 cohort。
- 商品切片：`head`/`longTail` 按训练期热度与 `longTailShare` 划分；`new` 为训练后才上架、模型无因子的商品，其相关对计入分母并记为未命中，不删除。
- 没有相关商品的验证用户列入 `noRelevantUsers` 并说明去向，不进入指标分母也不从总体中删除。
- 指标：宏平均 `recallAtK`、`ndcgAtK`（二值相关）、`coverage`（被推荐商品数/被评估用户候选空间并集）、流行度偏置（推荐商品的训练期热度均值与头部占比）；模型与基线同口径输出并给出差值。
- 目标函数：置信加权隐式 ALS，已曝光对按互动强度加权，已曝光未互动对软参与，未曝光对仅以均匀先验锚定；不做负采样。

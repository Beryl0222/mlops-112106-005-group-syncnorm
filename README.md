# 分组同步归一化

samples/scene.json 是本地合成的训练生产者和推理消费者。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行全部工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 replay_step.py samples/scene.json
python3 -m unittest discover -s tests -v
```

`requirements.txt` 只固定 NumPy；`replay_step.py` 与 `group_sync_norm` 只依赖标准库和 NumPy。

## 分段同步 BatchNorm 算子

`group_sync_norm` 把按校准组同步的 BatchNorm 做成可同时接入训练与推理的张量算子。协调端只与各分区交换按组、按通道的充分统计量；原始帧不出分区，也不存在“把各进程原始帧拼成整批”的旁路。

```python
from group_sync_norm import GroupNormTrainer, WorkerFrames

trainer = GroupNormTrainer(channels=2, epsilon=1e-5, decay=0.85, min_count=3)
trainer.seed_running({"global": {"mean": [0, 0], "var": [1, 2]}})

step = trainer.train_step(
    workers={"P0": WorkerFrames(x0, mask0, groups0, dy0),
             "P1": WorkerFrames(x1, mask1, groups1, dy1)},
    expected_workers=["P0", "P1"],
    groups=["large", "rare", "unseen"],
    gamma=gamma, beta=beta,
)
grads = trainer.backward_last(workers={"P0": dy0, "P1": dy1}, gamma=gamma)
trainer.promote_last()                       # 仅当本步被批准

call = trainer.open_inference("req-1")       # 冻结当前已批准统计量
y = call.normalize("rare", x, gamma, beta, epsilon=1e-5)
```

### 一步的四个阶段

| 阶段 | 内容 | 每个工作者每组每通道 |
| --- | --- | --- |
| `propose` | 各分区上报本地极值，合并成一个共享的 `shift/scale` 框架 | 2 |
| `reduce` | 各分区只归约自己的有效行，上报 `count/nonfinite/mean/m2` | 2 |
| `context` | 协调端广播合并后的 `(mean_q, inv_std_q)`，供所有分区前向使用 | 2 |
| `grad` | 各分区上报 `sum_g/sum_gq`，合并出 `dgamma/dbeta` 与 BN 链式项 | 2 |

`scale` 取覆盖合并极差的 2 的幂，除法精确，因此 1e8 附近的偏移不会把相近样本的方差抹平成 0。合并使用 Chan 并行公式，结果与“一次性归约全部行”在数学上相同，换分片不改变前向与梯度。

### 空组、掩码与异常

- **掩码行**：有效掩码之外的行（填充）不计入矩、不产生梯度，输出为零。
- **非有限有效行**：未被掩码但含 NaN/Inf 的行按 `nonfinite` 计入诊断，不参与归约与梯度；本步不会被批准，但其余组照常合并。
- **约定空组**：分区声明了组却没有任何有效帧，会上报 `none`；这是合法事件，其它分区不会等待。
- **未送达的分区**：根本不产生任何消息，由 `IncompleteExchange` 报告缺失的名字与组，而不是无限等待。
- **数值范围分歧**：若某分片归约出的值落在全部分区共同声明的范围之外，触发 `RangeViolation`，该分片本轮弃权且本步不可晋升。

### 何时可以给出前向值

当前向值只依赖该行自身与合并后的统计量时即可产出。一个组只要合并后 `count > 0` 就有有效输出；`count == 0` 的空组没有有效输出（该组行输出 0）。某个分区没有某组的输入不影响其他分区或其它组的推进。

### 统计量晋升与推理回退

- 只有**有效优化步**且同次运行统计量被接受时，整步统计量才晋升；否则整步丢弃（`promote_last()` 返回 `None`）。步被接受但没有组达到更新条件时，返回 `advanced` 为空的报告，便于区分“被拒绝”和“已接受但无条件可更新”。
- 已批准步中，有效样本数达到 `max(min_count, 2)` 的组提交均值与**样本方差**候选，按 `decay` 惯性更新；样本方差需要两次观测，所以单帧组即使可训练也不会覆盖服务端已有方差。首个候选没有历史时直接存入。
- 未达门槛的组**保持此前已批准的值不变**（既不被不合格候选拖低，也不被清零）。
- `global` 汇总所有有效帧，使用同样的门槛与惯性，是所有无自有条目的组的回退。
- 尚无任何批准的 `global` 时，请求未批准组会抛 `NotInferable`，即按推理策略报不可推理。

### 一致处置

梯度溢出、上游宣布跳步、内容冲突的重复分片，都会让本步的更新与统计量一起作废：已晋升的会回滚（`reject_pending()` / `reject_upstream()`），未晋升的不会再晋升。内容完全相同的重复分片视为重投，直接丢弃。

### 推理快照

`open_inference()` 在开始读取时冻结已批准表。之后的晋升或回滚都不会改变已开始调用读取的参数；新调用才看得到新版本。`explain_inference(group)` 说明某组走自有条目、回退 `global`，还是不可推理。

### 重放与排查

```sh
python3 replay_step.py samples/scene.json                     # 单分片 vs 逐行分片
python3 replay_step.py samples/rebalanced-group.json --splits 3,5
python3 replay_step.py samples/scene.json --group rare        # 为何可训练却没更新推理值
python3 replay_step.py samples/scene.json --json
```

`replay_step.py` 按行重新分片后重放同一步，报告前向、`dx`、`dgamma`、`dbeta` 的最大差异；`--group` 输出该组的有效帧数、门槛、是否晋升以及原因。

### 协议限制

- 交换内容只有按组、按通道的充分统计量，绝不含原始帧；`last_wire` 里可以逐条核对。
- 中间量预算由 `budget_bytes` 约束，`trainer.intermediate_bytes` 报告本步实际持有的临时数组（统计量、协调框架、行索引），不含调用方原始数据、输出张量或已批准统计小表。
- `tests/test_group_sync_norm.py` 覆盖：与单进程 FP64 参考的前向/梯度对比、换分片不变性、极端动态范围、空组/掩码/非有限区分、审批与回退、溢出与重复分片的回滚、推理快照、协议预算。

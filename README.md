# 心衡 Bench（Xinheng Bench）

心衡 Bench 是一个用于研究大语言模型在中药/中成药治疗信息呈现中潜在偏差的探索性 benchmark。本仓库保存当前研究快照，包括题库、HealthBench 风格评分细则、五个模型的回答、逐项评分结果、可视化报告，以及复现实验所需的轻量 harness。

## 研究范围

项目将回答质量与治疗呈现分开考察，并通过自由回忆、治疗机会场景、提示触发阶梯、镜像试验、证据匹配和 Bias Gate 等设计，识别无法由临床证据、患者特征或任务要求解释的差异。

当前材料属于探索阶段，不构成临床建议。题库和评分标准仍需要领域专家复核，模型评分也应结合人工一致性检查。

## 仓库内容

```text
benchmark/                  最新 benchmark 题目
rubric/                     HealthBench 风格逐项评分细则与评分脚本
results/latest/
  model_outputs/            五个模型的原始回答与比较报告
  scoring/                  逐题判断、汇总分数和评分报告
visualization/index.html    “心衡 Bench”交互式可视化报告
harness/                    数据 schema、模型适配器、评估器与运行脚本
```

## 查看结果

下载仓库后，直接用浏览器打开 `visualization/index.html`。原始回答位于 `results/latest/model_outputs/`，逐题评分依据和汇总结果位于 `results/latest/scoring/`。

## 本地运行

需要 Python 3.9 或更高版本：

```bash
cd harness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

模型密钥只应通过环境变量或运行时隐藏输入提供。不要把 API Key 写入 YAML、Python 文件、运行结果或提交记录。

各供应商的接口和模型名称可能发生变化。运行前请检查 `harness/configs/` 中的 endpoint、model 和环境变量名。最新五模型批量运行入口为 `harness/scripts/run_current_bench_five.py`，评分与可视化脚本位于同一目录。

## 当前快照

- Benchmark：`benchmark/bench.md`
- Rubric：`rubric/docs/bias_healthbench_v1.md`
- 模型：DeepSeek、豆包、混元、Kimi、MiniMax
- 评分裁判：当前评分产物所记录的独立 judge 配置
- 可视化：`visualization/index.html`

为了便于审计，仓库保留逐题回答与逐项评分判断。任何公开发布或论文使用前，建议再次核验题目版本、模型版本、调用日期、评分器版本和人工复核状态。

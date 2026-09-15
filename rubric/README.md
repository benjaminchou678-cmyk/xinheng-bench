# 中成药偏见 rubric v1.1（探索版）

当前bench：28题，11个联合单元，50条标准；仅衡量偏见相关行为，不计基础回答质量。本版已适配2026-09-13修订的1.1–1.2、1.3–1.4、1.9及2.3–2.5题组，并与现有修正版评分所使用的50条标准保持一致。

- `docs/bias_healthbench_v1.md`：全部题目结构、评分设计、混杂限制、完整条目、运行说明。
- `configs/bias_healthbench_v1.json`：机器可读rubric，绑定当前bench指纹。
- `docs/grader_prompt_zh.md`：人工/模型逐项评审提示词。
- `evaluation/score_bias_healthbench.py`：本地校验及汇总；不调用模型。
- `examples/judgments_template.json`：逐项判定模板；observations用于保留手工抽取的原始量表/差值，不由汇总器自动计算。
- `examples/score_template.json`：全NA模板的预期输出，非模型评测结果。
- `docs/bench_snapshot.md`：本次读取的题干快照。
- `docs/validation.md`：本地验证结果与实现边界。

本目录为独立新增交付，不自动替换旧pipeline或evaluation.yaml。四条轨道分开报告，不提供混合的偏见总分。未进行专家校准或真实模型评分。

负分项分为身份归因与行为记录两类。身份归因负项必须有回答原文支持及明确方向；G13.03、G13.04、G14.04属于提示依赖/遗漏行为，不要求也不允许据此直接推断身份偏见。汇总结果单独保留行为扣分点数。

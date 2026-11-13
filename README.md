# DAWL — Network Framework

精简的DAWL网络结构与核心脚本仓库，面向迁移到另一个项目／训练框架。

**这里只发布代码框架。** 不包含本地实验目录、数据、预训练权重、预测缓存、结果、日志或历史checkpoint。本地原项目未被移动或删除。

## 包含内容

- **ControlNet**：结构化ego／agent／lane编码，两层128维四头Transformer，静态attention与gate残差，30步控制输出。
- **WeightNet**：独立场景网络，八维logits、有界K8投影、单向encoder EMA。
- **物理链**：绝对前轮角可行映射、Euler自行车模型、30×0.1s rollout、带LiDAR杠杆臂的六点轨迹读出。
- **权重学习核心**：FP64隐式投影adjoint、D0困难度、显式NoSC证明接口、cost EMA。
- **主要脚本**：模型检查、准备好输入后的轨迹推理、训练集成入口、原评估器适配入口。
- 小型CPU软件测试。

完整的原K8／18 fixed成本、数据适配／标定、Theseus inner64、R5安全参数搜索和完整WAL事务需要由目标项目接入。它们没有被toy loss或假teacher替代；没有真实后端时训练入口会拒绝执行。详见[发布范围](docs/EXPORT_SCOPE.md)和[集成说明](docs/INTEGRATION.md)。

## 安装与快速检查

Python 3.9或更高版本；依赖PyTorch、NumPy。该导出首先在原研究环境Python3.9／PyTorch1.13的CPU路径验证，不构成所有新框架／设备的数值等价认证。

```bash
python -m pip install -e .
python scripts/model_smoke.py --device cpu
python scripts/train.py --inspect --supervision theseus_gt
python scripts/train.py --inspect --supervision difficulty_nosc
python scripts/evaluate.py --inspect
python -m unittest discover -s tests -v
```

model_smoke仅使用人工输入及随机权重检查结构，不是训练、数据集评价或安全结果。

## 目录

```text
src/dawl_core/
  models/       # ControlNet、WeightNet、初始化、encoder EMA
  physics/      # 映射、动力学、rollout、LiDAR读出
  weighting/    # 有界投影、困难度、NoSC、history
  interfaces.py # CostBackend／ExpertTeacher／ControlUpdatePolicy／事务接口
  training.py   # 原Control目标聚合、专家VJP及双梯度／提交辅助
scripts/
  model_smoke.py
  infer.py
  train.py
  evaluate.py
configs/model.json
tests/
docs/
```

## 模型与输入

外部接口的mask统一True=有效；ControlNet底层actor mask使用True=padding，planner负责边界转换。WeightNet直接接收valid mask。

|输入|形状|
|---|---|
|ego|B×10|
|ped／veh|B×N×6|
|有序左右lane|B×2×L×2|
|static|B×Ns×12|

Control输出ego B×30×2及agent B×N×30×2控制，随后经物理映射与积分。Weight输出B×8 logits，投影后sum(w)=1、w∈[.00625,.25]；倍率r=8w，范围[.05,2]。

```python
from dawl_core.models import build_control, build_weight
from dawl_core.physics import predict_trajectory
from dawl_core.weighting import project_weights

control = build_control(seed=27).eval()
weight = build_weight(seed=27).eval()
# scene由自己的adapter提供，raw物理量与features归一化视图分开。
# xy = predict_trajectory(control, scene)   # [B,6,2]，累计当前LiDAR XY米坐标
```

## 推理脚本

```bash
python scripts/infer.py \
  --input scene.npz \
  --checkpoint trusted_control_state_dict.pt \
  --output outputs/predictions.npz \
  --device cpu
```

输入要求见[场景schema](docs/SCENE_SCHEMA.md)。仓库不提供数据或权重。脚本只接收明确的Control tensor state_dict，不把历史完整训练checkpoint当作可自动恢复的模型；已有输出会拒绝覆盖。

## 两种Weight监督

共同Control目标为：
`L_C = mean(fixed_valid_sum + sum_valid(stop_gradient(r) * A))`。

- 专家引导：真实有限Theseus LM求解后用GT Outer生成dL/dr，只更新Weight允许的fusion/head；不向Control添加直接GT轨迹Loss。
- 困难度：使用已校准history、有效计数、rA相对困难度与熵正则；严格NoSC模式需要真实Jacobian／floor证明。

两组都有Control与Weight两个优化器。当前原实验中的R5是两组共同的额外Control更新规则；本精简仓库只定义其集成接口，不能将普通Adam命名成R5。

## 训练和评测接入

```bash
python scripts/train.py --supervision theseus_gt \
  --components your_project.integration:build_components \
  --config your_training.json

# 接口验证后，只有显式--run才调用用户实现的runner。
python scripts/train.py --supervision theseus_gt \
  --components your_project.integration:build_components \
  --config your_training.json --run
```

这些命令中的your_project是目标项目自备模块，不是仓库内隐藏的训练器。

evaluate.py只派发显式提供的原评估器adapter，不另造ACR／CCR。原研究中ACR=obj_box_col、CCR=用户HPC road_col扩展；CCR不是nuScenes原生服务器榜单项。

## 版本与来源

该导出来自DAWL冻结R5设计的代码子集，经包路径和工程耦合清理形成新的轻量版本；它不是原正式训练源码身份的替代品，也不继承旧Run、许可或性能结论。结构及边界说明见[架构](docs/ARCHITECTURE.md)，源归属见[第三方说明](THIRD_PARTY_NOTICES.md)。

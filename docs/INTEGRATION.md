# Integration contract

## Required scientific backends

- CostBackend.evaluate: 返回原顺序K8 A、8维valid、18 fixed、18维fixed_valid。A中base权重只计一次；未知可为NaN且valid=False，不当安全零。
- ExpertTeacher.evaluate: 只在真实LM完成后读取独立OuterTargets，返回每样本Outer和dL/dr。初始Control q及scene detach；真实有限3步LM／inner64／接受分支和cast guards由后端保证。
- ControlUpdatePolicy: 显式定义Control候选与提交；完整原R5必须保留all74-before20 clip、None-skip、真实native moment及候选参数覆盖。仓库不提供普通Adam冒充R5。
- CompleteStateTransaction: 从forward前开始覆盖双网络、双Adam、scheduler、history、EMA、sampler、modes、CPU/CUDA RNG和attempt计数。INTENT/RETURN及崩溃UNKNOWN不能由空实现冒充。

TrainingComponents.validate只验证接口存在，不是数值、来源、物理或科学资格认证。实际数据runner由目标项目提供。导入plugin会执行目标项目的可信Python代码。

## Same-pre-update gradients

training.prepare_joint_gradients只计算梯度，不执行optimizer：
1. 同一前状态先建立Control／Weight两个loss。
2. backward(Control)后Weight grad必须None。
3. backward(Weight)后Control梯度必须与先前exact相同。
4. 全局归约在clip之前；Control全74 clip5，Weight允许组clip1。

commit_joint_update要求已开始的完整事务，按Control→Weight提交，成功后才执行scheduler／cached history／EMA等callbacks。真实R5的候选参数与Adam moments关系由policy实现。不在Control step之后重新计算Weight梯度。不保证没有外部事务实现时的原子恢复。

## Difficulty / NoSC

CostHistory必须用训练Core而非验证集统计初始化。D0缓存的是更新前rA，成功后按原valid计数commit，beta.99。NoSC只在真实floor与Jacobian每元素严格零的证明通过时排除SC Weight监督，分母仍含全部原cost-valid cell；Control SC、历史SC、八维熵都保留。sc_certificate缺失不能宣称使用了严格NoSC。

## Evaluation

目标项目应导出float32累计当前LiDAR XY[N,6,2]及token身份，交给原SparseDrive评估器。ACR严格等于其obj_box_col（原GT-self mask及车框规则）；CCR需要原用户HPC扩展。仓库不以新自定义公式填补缺失指标。

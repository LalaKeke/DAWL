# Architecture and algorithm boundaries

## ControlNet

- ego: 10→128, LayerNorm, ReLU, 128→128。
- pedestrian／vehicle: each6→64→128。
- map point: 2→32→128；左右边界展平为2L个token。
- 2-layer TransformerEncoder: d_model128, heads4, FFN2048, ReLU, dropout.1, post-norm, LayerNorm eps1e-5。
- token residual MLP128→128→128。
- static: 12→64→128→LayerNorm，attention池化＋sigmoid风险gate，再形成ego残差。
- ego／ped／veh各自128→128→60 Tanh控制头，默认30步。

Control总1,390,806参数；Weight总1,335,594参数。数字描述当前架构，不是数据集效果。

## WeightNet

保留独立场景融合和静态残差，仅ego token接128→128→8 decoder。五个encoder由单向Control→Weight EMA拥有，fusion/head梯度训练。scratch_then_ema，不在构造时暗中复制Control。factory的初始化优先于类构造默认；跨框架同seed不保证初始张量逐值相同。

投影采用原半圆有界simplex映射：sum(w)=1，.00625≤w≤.25，r=8w。隐式根解／adjoint使用FP64，forward保留输入dtype及原拒绝规则。不是softmax+clip或STE。

## Physical readout

Tanh输出先解码为加速度与绝对前轮角，再按原ay≤3、steering-rate≤.4、wheelbase2.588／dt.1映射和Euler展开。状态为[x,y,yaw,v]。取索引4/9/14/19/24/29，带实际标定和yaw旋转杠杆臂转为未来LiDAR传感器原点六点XY。未知／不可行条件不伪造为成功。

## Training boundary

专家机制依赖真实有限3步LM inner64，不是收敛argmin隐式梯度，也不是GT直接监督Control。困难度机制依赖真实K8代价、原有效分母和history，不代表loss数值大就一定有安全收益。

完整R5／旧十八fixed／dataset预处理不在本仓库。保留接口是为了让目标项目明确接入真实后端。没有原实验复现或部署安全声明。

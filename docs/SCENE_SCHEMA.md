# Scene NPZ input for scripts/infer.py

以allow_pickle=False读取。字段必须与load_scene中的完整字段集相符；不接收GT future或额外不明字段。

|字段|dtype / shape|
|---|---|
|features_ego|float32 B×10|
|features_ped|float32 B×Np×6|
|features_veh|float32 B×Nv×6|
|features_lane|float32 B×2×L×2|
|features_static|float32 B×Ns×12|
|raw_ego_state|float32 B×10，实际物理状态／路线条件|
|raw_ped_states / raw_veh_states|float32 B×N×6|
|raw_ped_mask / raw_veh_mask|bool B×N，True=有效|
|raw_static_mask|bool B×Ns|
|raw_map_mask|bool B×2×L|
|raw_previous_steering|float32 B|
|previous_steering_source|string scalar，estimated或assumed；每批同一来源|
|control_to_lidar|float32 B×4×4|
|sensor_offset|float32 B×3|
|tokens|string B，唯一token|

features由目标adapter完成真实归一化；raw不得被神经特征归一化覆盖。当前框架不自行读取CAN、N3、SparseDrive缓存或旧训练统计。lane是有序左右边界，不能随意换成未关联polyline。传感器杠杆臂与初始转角不能用无来源常数假造。

输出NPZ含tokens与trajectories，后者为float32 B×6×2，时刻0.5、1、1.5、2、2.5、3秒；是累计位置而非逐步位移。导出本身不构成官方测评。

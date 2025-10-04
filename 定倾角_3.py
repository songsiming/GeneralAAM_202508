"""
迭代说明：
	2025-09-29:
		1) 基于“三维_比例_9.py”，使用“虚拟指令加速度”+PID，修改而来
		2) 实现PID自动调参
		3) 测试多种态势
迭代日期：2025-09-26
涉及到的坐标系：
	弹体坐标系(B系)：
		O->导弹的质心
		X->弹体轴线方向，指向弹头部
		Y->位于弹体纵向对称面内与X轴垂直，指向弹体上方
		Z->按右手直角坐标系确定，指向弹体右侧
	速度坐标系(V系):
		O>导弹的质心
		X->来流方向，与速度矢量V重合（大致指向弹头部）
		Y->弹体纵向对称面内与X垂直（大致指向弹体上方）
		Z->按右手直角坐标系确定（大致指向弹体右侧）
	导航坐标系(N系)：
		O->导弹的质心
		X->北
		Y->天
		Z->东
	地理坐标系(G系):
		O->导弹发射位置
		X->北
		Y->天
		Z->东
"""
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

# 中文字符不报错
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


# 对发动机仿真
def missile_engine(missile_engine_mass_t0, sim_time):
	first_stage_thrust = 14000  # 一阶段推力
	first_mass_consumption = 1.8  # 一阶段质量消耗流率
	first_work_time = 6  # 一阶段工作时间
	second_stage_thrust = 8000  # 二阶段推力
	second_mass_consumption = 0.9  # 二阶段质量消耗流率
	second_work_time = 8  # 二阶段工作时间

	# 第一级发动机工作时间段
	if sim_time <= first_work_time:
		# 第一级工作：消耗燃料并产生推力
		missile_engine_mass = missile_engine_mass_t0 - first_mass_consumption * sim_time
		missile_engine_thrust = first_stage_thrust
	# 第二级发动机工作时间段
	elif sim_time <= first_work_time + second_work_time:
		# 第二级工作：消耗燃料并产生推力
		# 计算第一级结束时的质量
		first_stage_end_mass = missile_engine_mass_t0 - first_mass_consumption * first_work_time
		# 计算第二级工作时间
		second_stage_working_time = sim_time - first_work_time
		missile_engine_mass = first_stage_end_mass - second_mass_consumption * second_stage_working_time
		missile_engine_thrust = second_stage_thrust
	# 发动机关闭
	else:
		# 计算总工作结束时的质量
		first_stage_end_mass = missile_engine_mass_t0 - first_mass_consumption * first_work_time
		missile_engine_mass = first_stage_end_mass - second_mass_consumption * second_work_time
		missile_engine_thrust = 0

	return missile_engine_mass, missile_engine_thrust


# 三维比例导引
def proportional_guidance(fix_v=0.0, fix_h=0.0):
	plt_yes = False
	if fix_h != 0:
		plt_yes = True

	# 态势参数
	missile_pos = np.array([0.0, 0.0, 0.0])  # 导弹初始位置(北-天-东) [m]
	missile_vel = np.array([250.0, 0.0, 0.0])  # 导弹初始速度(北-天-东) [m/s]
	target_pos = np.array([76000.0, 0.0, 0.0])  # 目标初始位置(北-天-东) [m]
	target_vel = np.array([0, 20, -160])  # 目标初始速度(北-天-东) [m/s]

	# 定倾角弹道参数
	missile_gamma = np.arctan2(missile_vel[1], missile_vel[0])
	given_gamma = np.deg2rad(18)  # 给定弹道倾角
	gamma_loft = 5000  # 转平飞高度
	integral_err_gamma = 0.0
	prev_err_gamma = 0.0
	Kp_gamma = 2000
	Ki_gamma = 0.0001
	Kd_gamma = 0.0001
	missile_gamma_max = 0.0
	gamma_control_yes = 0.0

	# 仿真参数
	dt = 0.01  # 仿真时间步长 [s]
	t_total = 100.0  # 总仿真时间 [s]
	plt_stop = 150  # 绘图时提前结束的步数

	# 导弹固有参数
	missile_body_d = 0.178  # 导弹直径 [m]
	missile_load_l = 1.65  # 导弹载荷段长度 [m]
	missile_load_mass = 90  # 导弹载荷重量(除燃料外的导弹空重)
	missile_engine_l = 2.00  # 导弹动力段长度 [m]
	missile_engine_mass_t0 = 67  # 导弹发动机质量(初始质量)
	mt_dist_lock_delta = 200  # 末制导锁定舵偏角的距离 [m]
	N_v = 4.0  # 比例导引系数(铅锤方向)
	K_delta = 1.0  # 一阶惯性环节，稳态增益(铅锤方向)
	tau = 0.008  # 一阶惯性环节，时间常数(舵机响应时间的数量级通常在几ms至几十ms)(铅锤方向) [s]
	delta_e_max = np.deg2rad(25)  # 升降舵偏角最大值(铅锤方向) [rad]
	delta_e_dot_max = np.deg2rad(300)  # 升降舵角速率最大值(铅锤方向) [rad/s]
	Kp_v = 0.01  # PID系数(铅锤方向)
	Ki_v = 0.0001  # PID系数(铅锤方向)
	Kd_v = 0.001  # PID系数(铅锤方向)
	missile_q_max = np.deg2rad(15)  # 俯仰角速率最大值(铅锤方向) [rad/s]
	N_h = 4.0  # 比例导引系数(水平方向)
	delta_r_max = np.deg2rad(48)  # 方向舵偏角最大值(水平方向) [rad]
	delta_r_dot_max = np.deg2rad(600)  # 方向舵角速率最大值(水平方向) [rad/s]
	Kp_h = 0.00024  # PID系数(水平方向) 调试时，这个数值在e-4的量级，打不中就稍微调大一点
	Ki_h = 0.000015  # PID系数(水平方向)
	Kd_h = 0.000001  # PID系数(水平方向)
	omega_n = 100  # 固有频率，较高的频率代表快速响应(水平方向) [rad/s]
	zeta = 0.7  # 欠阻尼，有轻微超调(水平方向)
	missile_r_max = np.deg2rad(25)  # 偏航角速率最大值(水平方向) [rad/s]
	C_L_alpha = 3.0  # 气动相关，升力系数斜率 [1/rad]
	C_C_beta = -3.5  # 气动相关，侧力系数斜率 [1/rad]
	K_ind = 0.05  # 气动相关，诱导阻力因子，没有一般值 (建议0.05-0.15之间)
	C_D_0 = 0.05  # 气动相关，零升阻力系数(即寄生阻力系数，来自机体摩擦阻力、表面摩擦、形状阻力，和升力、侧力无关)
	C_m_delta_e = 3.0  # 气动相关，俯仰力矩系数对舵偏角导数(舵效系数) [1/rad] TODO: 取-1.7至-4.6[/rad]或-0.03至-0.08
	C_m_delta_r = 3.0  # 气动相关，偏航力矩系数对舵偏角导数(舵效系数) [1/rad] TODO: 取-1.7至-4.6[/rad]或-0.03至-0.08
	C_m_alpha = -0.8  # 气动相关，俯仰力矩系数对迎角导数(静稳定性导数) [1/rad] TODO: 取-8.6至-28.6[/rad]或-0.15至-0.20
	C_m_beta = -0.8  # 气动相关，偏航力矩系数对侧滑角导数(静稳定性导数) [1/rad] TODO: 取-8.6至-28.6[/rad]或-0.15至-0.20
	C_m_q = -0.2  # 气动相关，俯仰(纵向)阻尼导数 TODO: 取-0.2至-1.5[/rad]
	C_m_r = -0.2  # 气动相关，偏航(横航向)阻尼导数 TODO: 取-0.2至-1.5[/rad]

	# 一些需要赋初值的变量
	missile_alpha = 0  # 初始时刻，导弹迎角 [rad]
	missile_beta = 0  # 初始时刻，导弹侧滑角 [rad]
	acc_b_y = 0  # 初始时刻，实际法向过载(垂直弹轴向上)(铅锤方向) [m/s^2]
	acc_b_z = 0  # 初始时刻，实际法向过载(垂直弹轴向右)(水平方向) [m/s^2]
	err_v_int = 0.0  # 初始时刻，误差积分(铅锤方向)
	err_v_prev = 0.0  # 初始时刻，误差(铅锤方向)
	err_h_int = 0.0  # 初始时刻，误差积分(水平方向)
	err_h_prev = 0.0  # 初始时刻，误差(水平方向)

	# 解析一些变量
	missile_mass_t0 = missile_load_mass + missile_engine_mass_t0  # 导弹质量 [kg]
	missile_body_l = missile_load_l + missile_engine_l  # 导弹长度 [m]
	Kp_h_user = Kp_h  # PID系数(水平方向，计算使用)
	n_steps = int(t_total / dt)  # 总步数
	stop_i = n_steps  # 声明，仿真结束时刻
	missile_body_r = missile_body_d / 2  # 导弹半径 [m]
	rel_pos_t0 = target_pos - missile_pos  # 初始时刻相对位置
	S_ref = np.pi * missile_body_r * missile_body_r  # 参考面积 [m^2]
	missile_I_yy = (missile_mass_t0 / 12) * (3 * missile_body_r * missile_body_r + missile_body_l * missile_body_l)  # 转动惯量(俯仰惯量)【其实俯仰方向转动惯量，也应该参考偏航方向计算】
	missile_I_zz_1 = (missile_mass_t0 / 12) * (3 * missile_body_r * missile_body_r + missile_body_l * missile_body_l)  # 转动惯量(偏航惯量)
	missile_I_zz_1 = 1.0 * missile_I_zz_1
	# 【偏航方向，考虑多段】计算各段绕自身质心的转动惯量
	Izz_engine_local = (missile_engine_mass_t0 / 12) * (3 * missile_body_r * missile_body_r + missile_engine_l * missile_engine_l)
	Izz_load_local = (missile_load_mass / 12) * (3 * missile_body_r * missile_body_r + missile_load_l * missile_load_l)
	# 【偏航方向，考虑多段】计算总质心位置（从导弹尾部开始计算）后半段质心位置
	engine_cg = missile_engine_l / 2
	# 【偏航方向，考虑多段】计算总质心位置（从导弹尾部开始计算）前半段质心位置
	load_cg = missile_engine_l + missile_load_l / 2
	# 【偏航方向，考虑多段】总质心位置
	total_cg = (missile_engine_mass_t0 * engine_cg + missile_load_mass * load_cg) / missile_mass_t0
	# 【偏航方向，考虑多段】计算各段质心到总质心的距离，及总质心位置
	d_engine = abs(engine_cg - total_cg)
	d_load = abs(load_cg - total_cg)
	# 【偏航方向，考虑多段】使用平行轴定理计算总转动惯量
	missile_I_zz_t0 = (Izz_engine_local + missile_engine_mass_t0 * d_engine * d_engine + Izz_load_local + missile_load_mass * d_load * d_load)

	# 上一时刻参数（顺序：舵面->姿态->速度->位置，T在前M在后）
	delta_e_prev = 0.0  # 上一时刻，升降舵偏角 [rad]
	delta_r_prev = 0.0  # 上一时刻，方向舵偏角 [rad]
	delta_e_dot_prev = 0.0  # 上一时刻，上一次舵偏角速率(铅锤方向) [1/rad]
	delta_r_dot_prev = 0.0  # 上一时刻，上一次舵偏角速率(水平方向) [1/rad]
	missile_q_prev = 0.0  # 上一时刻，俯仰角速率 [rad/s]
	missile_r_prev = 0.0  # 上一时刻，偏航角速率 [rad/s]
	missile_theta_prev = np.arctan2(missile_vel[1], missile_vel[0])  # 上一时刻，导弹俯仰角 [rad]
	missile_psi_prev = np.arctan2(-missile_vel[2], missile_vel[0])  # 上一时刻，导弹俯仰角 [rad]
	lambda_v_prev = np.arctan2(rel_pos_t0[1] + fix_v, rel_pos_t0[0])  # 上一时刻，铅锤方向视线角 [rad]
	lambda_h_prev = np.arctan2(-rel_pos_t0[2] - fix_h, rel_pos_t0[0])  # 上一时刻，水平方向视线角 [rad]
	vel_n_x_prev = missile_vel[0]  # 上一时刻，N系导弹速度(北向) [m/s]
	vel_n_y_prev = missile_vel[1]  # 上一时刻，N系导弹速度(天向) [m/s]
	vel_n_z_prev = missile_vel[2]  # 上一时刻，N系导弹速度(东向) [m/s]
	pos_g_x_prev = missile_pos[0]  # 上一时刻，G系导弹位置(北向) [m]
	pos_g_y_prev = missile_pos[1]  # 上一时刻，G系导弹位置(天向) [m]
	pos_g_z_prev = missile_pos[2]  # 上一时刻，G系导弹位置(东向) [m]

	# 环境参数
	rho = 1.225  # 空气密度 [kg/m^3]
	g = 9.81  # 重力加速度 [m/s^2]

	# 存储数据（顺序：舵面->姿态->速度->位置，T在前M在后）
	target_pos_history = np.zeros((n_steps, 3))  # 存储目标位置(G系)
	err_h_history = np.zeros(n_steps)  # 存储横向误差
	delta_e_history = np.zeros(n_steps)  # 存储舵偏角
	delta_r_history = np.zeros(n_steps)  # 存储舵偏角
	missile_mass_history = np.zeros(n_steps)  # 存储导弹质量
	missile_thrust_history = np.zeros(n_steps)  # 存储导弹推力
	missile_theta_history = np.zeros(n_steps)  # 存储导弹俯仰角
	missile_gamma_history = np.zeros(n_steps)  # 存储导弹弹道倾角
	missile_alpha_history = np.zeros(n_steps)  # 存储导弹迎角
	missile_psi_history = np.zeros(n_steps)  # 存储导弹偏航角
	missile_psis_history = np.zeros(n_steps)  # 存储导弹弹道偏角
	missile_beta_history = np.zeros(n_steps)  # 存储导弹侧滑角
	missile_lift_history = np.zeros(n_steps)  # 存储导弹升力
	missile_drag_history = np.zeros(n_steps)  # 存储导弹阻力
	missile_side_history = np.zeros(n_steps)  # 存储导弹侧力
	missile_speed_history = np.zeros(n_steps)  # 存储速度大小
	missile_vel_history = np.zeros((n_steps, 3))  # 存储速度矢量
	missile_pos_history = np.zeros((n_steps, 3))  # 存储导弹位置(G系)
	mt_dist_history = np.zeros(n_steps)  # 存储导弹与目标距离
	missile_nx_history = np.zeros(n_steps)  # 存储过载大小
	missile_ny_history = np.zeros(n_steps)  # 存储过载大小
	missile_nz_history = np.zeros(n_steps)  # 存储过载大小
	missile_lambda_history = np.zeros((n_steps, 2))  # 存储视线角
	missile_accC_history = np.zeros((n_steps, 2))  # 存储指令加速度
	missile_acc1s_history = np.zeros(n_steps)  # 存储加速度大小(标量)
	missile_acc1v_history = np.zeros((n_steps, 3))  # 存储加速度大小(矢量)
	missile_acc2s_history = np.zeros(n_steps)  # 存储弹道坐标系加速度(标量)
	missile_acc2v_history = np.zeros((n_steps, 3))  # 存储弹道坐标系加速度(矢量)
	missile_acc2C_history = np.zeros((n_steps, 2))  # 存储弹道坐标系指令加速度(矢量)
	missile_accNs_history = np.zeros(n_steps)  # 存储导航坐标系加速度(标量)
	missile_accNv_history = np.zeros((n_steps, 3))  # 存储导航坐标系加速度(矢量)

	# 打印
	if missile_I_zz_1 > missile_I_zz_t0 and plt_yes:
		print('missile_I_zz fix success!')
	else:
		print('missile_I_zz fix failed!')

	# 循环开始
	for i in range(n_steps):
		# 更新目标状态（假设目标匀速直线运动）
		target_pos = target_pos + target_vel * dt

		# 计算相对位置向量 (导弹指向目标)
		rel_pos = target_pos - missile_pos
		mt_dist = np.linalg.norm(rel_pos)  # 当前距离
		mt_dist_history[i] = mt_dist

		# 终止条件：如果距离小于5m，则认为命中
		if mt_dist < 5:
			print(f"命中目标于时间 t = {i * dt:.2f} s mt_dist = {mt_dist:.2f}")
			# 截断记录数组
			target_pos_history = target_pos_history[:i]
			missile_pos_history = missile_pos_history[:i]
			missile_acc1v_history = missile_acc1v_history[:i]
			mt_dist_history = mt_dist_history[:i]
			stop_i = i
			break
		# 终止条件：脱靶
		if mt_dist > mt_dist_history[i - 2] and i * dt > 10:
			mt_dist_v = np.linalg.norm([rel_pos[0], rel_pos[1]])
			mt_dist_h = np.linalg.norm([rel_pos[0], rel_pos[2]])
			if plt_yes:
				print(f"脱靶 t = {i * dt:.2f} s, 脱靶距离 = {mt_dist:.2f} m, 纵向脱靶距离={mt_dist_v:.2f}, 横向脱靶距离={mt_dist_h:.2f}, {rel_pos[1]:.2f}, {rel_pos[2]:.2f}")
			stop_i = i
			fix_v = rel_pos[1]
			fix_h = rel_pos[2]
			break

		# 计算视线角 (lambda) [rad]
		lambda_v = np.arctan2(rel_pos[1] + fix_v, rel_pos[0])  # 注意：arctan2(y, x) 返回的是与x轴的夹角
		lambda_h = np.arctan2(-rel_pos[2] - fix_h, rel_pos[0])  # 注意：arctan2(z, x) 返回的是与x轴的夹角(逆时针)

		# 计算视线角速率 (dLambda/dt) [rad/s] - 使用后向差分
		if i == 0:
			lambda_v_dot = 0.0
			lambda_h_dot = 0.0
		else:
			# 视线角变化量
			lambda_v_d = lambda_v - lambda_v_prev
			lambda_h_d = lambda_h - lambda_h_prev
			# 角度归一化，保证差值在 [-pi, pi]
			lambda_v_d = (lambda_v_d + np.pi) % (2 * np.pi) - np.pi
			lambda_h_d = (lambda_h_d + np.pi) % (2 * np.pi) - np.pi
			# 计算视线角速率
			lambda_v_dot = lambda_v_d / dt
			lambda_h_dot = lambda_h_d / dt
		lambda_v_prev = lambda_v  # 更新上一时刻视线角(铅锤方向）
		lambda_h_prev = lambda_h  # 更新上一时刻视线角(水平方向)

		# 计算当前导弹速度大小和弹道倾角(计算(水平/垂直)分量投影)
		V_m = np.linalg.norm(missile_vel)
		V_m_v = np.linalg.norm([missile_vel[0], missile_vel[1]])
		V_m_h = np.linalg.norm([missile_vel[0], missile_vel[2]])

		# 计算指令加速度 (ac) [m/s^2]
		acc_cmd_v = N_v * V_m_v * lambda_v_dot
		acc_cmd_h = N_h * V_m_h * lambda_h_dot

		# TODO: 在这里计算定倾角的指令加速度
		if missile_pos[1] > gamma_loft:
			given_gamma = 0
		if np.linalg.norm(rel_pos) > 20e3:
			# 计算误差
			err_gamma = missile_gamma - given_gamma
			err_gamma = -err_gamma

			# 计算PID各项
			# 比例项
			P_gamma = Kp_gamma * err_gamma

			# 积分项（指数衰减近似）
			forgetting_factor = 1.0 - (1.0 / 1000)  # 近似等效于500个点的窗口
			integral_err_gamma = integral_err_gamma * forgetting_factor + err_gamma * dt
			I_gamma = Ki_gamma * integral_err_gamma

			# 微分项（使用后向差分）
			derivative_err_gamma = (err_gamma - prev_err_gamma) / dt
			D_gamma = Kd_gamma * derivative_err_gamma
			prev_err_gamma = err_gamma

			# 合成PID输出
			acc_cmd_v = P_gamma + I_gamma + D_gamma
			acc_cmd_v = np.clip(acc_cmd_v, -20 * g, 20 * g)

		# 记录符合高度的步数
		if abs(missile_gamma - given_gamma) <= np.deg2rad(1):
			gamma_control_yes = gamma_control_yes + 1

		# 记录导弹最大高度(便于分析超调量)
		if missile_gamma > missile_gamma_max:
			missile_gamma_max = missile_gamma

		# 计算误差
		err_v = acc_cmd_v - acc_b_y
		err_h = acc_cmd_h + acc_b_z  # 这里改了个负号

		control_v_method = 2
		if control_v_method == 1:
			# (铅锤方向)根据误差计算升降舵偏角速率
			delta_e_dot = (K_delta * err_v - delta_e_prev) / tau
			if abs(delta_e_dot) > delta_e_dot_max:
				delta_e_dot = np.sign(delta_e_dot) * delta_e_dot_max
			# (铅锤方向)根据升降舵偏角速率计算升降舵偏角
			delta_e = delta_e_prev + delta_e_dot * dt
			if abs(delta_e) > delta_e_max:
				delta_e = np.sign(delta_e) * delta_e_max
		elif control_v_method == 2:
			# (铅锤方向)比例项更新(与当前误差成正比，大力出奇迹)
			P_v = Kp_v * err_v
			# (铅锤方向)积分项更新(与过去所有误差的积累成正比，消除残差)
			err_v_int += err_v * dt
			I_v = Ki_v * err_v_int
			# (铅锤方向)微分项(与误差的变化速率成正比，类似刹车或阻尼)
			err_v_der = (err_v - err_v_prev) / dt
			D_v = Kd_v * err_v_der
			# (铅锤方向)PID输出 = 舵令（假设线性关系）
			delta_e_cmd_pid = P_v + I_v + D_v
			# (铅锤方向)更新误差
			err_v_prev = err_v
			# (铅锤方向)加舵面限幅
			delta_e_cmd = np.clip(delta_e_cmd_pid, -delta_e_max, delta_e_max)
			# (铅锤方向)定义状态变量
			x1 = delta_e_prev
			x2 = delta_e_dot_prev
			# (铅锤方向)定义状态方程
			x1_dot = x2
			x2_dot = - omega_n * omega_n * x1 - 2 * zeta * omega_n * x2 + omega_n * omega_n * delta_e_cmd
			# (铅锤方向)欧拉积分，更新状态变量
			delta_e = x1 + x1_dot * dt
			delta_e_dot = x2 + x2_dot * dt
			# (铅锤方向)速度限幅 (独立于位置限幅)
			if abs(delta_e_dot) > delta_e_dot_max:
				delta_e_dot = np.sign(delta_e_dot) * delta_e_dot_max
			# (铅锤方向)速率限幅 (如果位置达到限幅，速度应置为零)【上一时刻已经达到或超过正限幅，并且当前速度是正的（还想继续往上超）】or【上一时刻已经达到或超过负限幅，并且当前速度是负的（还想继续往下超）】
			if (delta_e_prev >= delta_e_max and delta_e_dot > 0) or (delta_e_prev <= -delta_e_max and delta_e_dot < 0):
				delta_e_dot = 0.0
			# (铅锤方向)位置限幅【这是最后一道保险，保证数值积分误差不会导致最终结果超限】
			if abs(delta_e) > delta_e_max:
				delta_e = np.sign(delta_e) * delta_e_max
		elif control_v_method == 3:
			# (铅锤方向)比例项更新(与当前误差成正比，大力出奇迹)
			P_v = Kp_v * err_v
			# (铅锤方向)积分项更新(与过去所有误差的积累成正比，消除残差)
			err_v_int += err_v * dt
			I_v = Ki_v * err_v_int
			# (铅锤方向)微分项(与误差的变化速率成正比，类似刹车或阻尼)
			err_v_der = (err_v - err_v_prev) / dt
			D_v = Kd_v * err_v_der
			# (铅锤方向)PID输出 = 舵令（假设线性关系）
			delta_e_cmd_pid = P_v + I_v + D_v
			# (铅锤方向)更新误差
			err_v_prev = err_v
			# (铅锤方向)加舵面限幅
			delta_e_cmd = np.clip(delta_e_cmd_pid, -delta_e_max, delta_e_max)
			# (铅锤方向)计算舵偏角速率(一阶惯性环节)
			delta_e_dot = (K_delta * delta_e_cmd - delta_e_prev) / tau
			# (铅锤方向)计算舵偏角
			delta_e = delta_e_prev + delta_e_dot * dt
		else:
			# (铅锤方向)比例项更新(与当前误差成正比，大力出奇迹)
			P_v = Kp_v * err_v
			# (铅锤方向)积分项更新(与过去所有误差的积累成正比，消除残差)
			err_v_int += err_v * dt
			I_v = Ki_v * err_v_int
			# (铅锤方向)微分项(与误差的变化速率成正比，类似刹车或阻尼)
			err_v_der = (err_v - err_v_prev) / dt
			D_v = Kd_v * err_v_der
			# (铅锤方向)PID输出 = 舵令（假设线性关系）
			delta_e_cmd_pid = P_v + I_v + D_v
			# 理想
			delta_e = delta_e_cmd_pid
			delta_e_dot = 0

		# (水平方向)比例项更新(与当前误差成正比，大力出奇迹)
		P_h = Kp_h * err_h
		# (水平方向)积分项更新(与过去所有误差的积累成正比，消除残差)
		err_h_int += err_h * dt
		I_h = Ki_h * err_h_int
		# (水平方向)微分项(与误差的变化速率成正比，类似刹车或阻尼)
		err_h_der = (err_h - err_h_prev) / dt
		D_h = Kd_h * err_h_der
		# (水平方向)PID输出 = 舵令（假设线性关系）
		delta_r_cmd_pid = P_h + I_h + D_h
		# (水平方向)更新误差
		err_h_prev = err_h
		# (水平方向)加舵面限幅
		delta_r_cmd = np.clip(delta_r_cmd_pid, -delta_r_max, delta_r_max)
		# (水平方向)先算第2个状态方程(第1个状态方程太简单，具体参考笔记P12)
		delta_r_dot_dot = - omega_n * omega_n * delta_r_prev - 2 * zeta * omega_n * delta_r_dot_prev + omega_n * omega_n * delta_r_cmd
		# (水平方向)欧拉积分
		delta_r_dot = delta_r_dot_prev + delta_r_dot_dot * dt
		# (水平方向)速率限幅 (如果位置达到限幅，速度应置为零)
		if delta_r_prev > delta_r_max:
			delta_r_dot = 0.0
		elif delta_r_prev < -delta_r_max:
			delta_r_dot = 0.0
		# (水平方向)速度限幅 (独立于位置限幅)
		if delta_r_dot > delta_r_dot_max:
			delta_r_dot = delta_r_dot_max
		elif delta_r_dot < -delta_r_dot_max:
			delta_r_dot = -delta_r_dot_max
		delta_r = delta_r_prev + delta_r_dot * dt
		# (水平方向)位置限幅
		if delta_r > delta_r_max:
			delta_r = delta_r_max
		elif delta_r < -delta_r_max:
			delta_r = -delta_r_max
		# (水平方向)末制导调整PID系数
		if mt_dist < 2000:
			# 定义两个端点
			x = [1500, 2000]  # mt_dist值
			y = [0.0005, Kp_h_user]  # 对应的Kp值
			# 线性插值（注意：这里顺序是反的，因为mt_dist减小，Kp增加）
			Kp_h = np.interp(mt_dist, x, y)

		# 末制导锁死舵偏角
		if mt_dist < mt_dist_lock_delta:
			delta_e = delta_e_prev
			delta_r = delta_r_prev

		# 计算动压
		Q_v = 0.5 * rho * V_m_v * V_m_v
		Q_h = 0.5 * rho * V_m_h * V_m_h

		# 由动压合舵偏角计算控制力矩
		M_delta_e = Q_v * S_ref * missile_body_d * C_m_delta_e * delta_e
		M_delta_r = Q_h * S_ref * missile_body_d * C_m_delta_r * delta_r

		# 由迎角和侧滑角计算稳定性力矩
		M_alpha = Q_v * S_ref * missile_body_d * C_m_alpha * missile_alpha
		M_beta = Q_h * S_ref * missile_body_d * C_m_beta * missile_beta

		# 由俯仰角速率和偏航角速率计算阻尼力矩
		M_q = Q_v * S_ref * missile_body_d * C_m_q * missile_q_prev * missile_body_d / (2 * V_m_v)
		M_r = Q_h * S_ref * missile_body_d * C_m_r * missile_r_prev * missile_body_d / (2 * V_m_h)

		# 计算导弹质量
		missile_engine_mass_t, missile_engine_thrust_t = missile_engine(missile_engine_mass_t0, i * dt)
		missile_mass_t = missile_load_mass + missile_engine_mass_t

		# 计算Izz【偏航方向，考虑多段】计算各段绕自身质心的转动惯量
		Izz_engine_local = (missile_engine_mass_t / 12) * (3 * missile_body_r * missile_body_r + missile_engine_l * missile_engine_l)
		Izz_load_local = (missile_load_mass / 12) * (3 * missile_body_r * missile_body_r + missile_load_l * missile_load_l)
		# 【偏航方向，考虑多段】计算总质心位置（从导弹尾部开始计算）后半段质心位置
		engine_cg = missile_engine_l / 2
		# 【偏航方向，考虑多段】计算总质心位置（从导弹尾部开始计算）前半段质心位置
		load_cg = missile_engine_l + missile_load_l / 2
		# 【偏航方向，考虑多段】总质心位置
		total_cg = (missile_engine_mass_t * engine_cg + missile_load_mass * load_cg) / missile_mass_t
		# 【偏航方向，考虑多段】计算各段质心到总质心的距离，及总质心位置
		d_engine = abs(engine_cg - total_cg)
		d_load = abs(load_cg - total_cg)
		# 【偏航方向，考虑多段】使用平行轴定理计算总转动惯量
		missile_I_zz_t = (Izz_engine_local + missile_engine_mass_t * d_engine * d_engine + Izz_load_local + missile_load_mass * d_load * d_load)

		# 由合力矩计算俯仰角加速度
		M_total_v = M_delta_e + M_alpha + M_q
		M_total_h = M_delta_r + M_beta + M_r
		missile_q_dot = M_total_v / missile_I_yy
		missile_r_dot = M_total_h / missile_I_zz_t

		# (铅锤方向)由俯仰角加速度计算俯仰角速度
		missile_q = missile_q_prev + missile_q_dot * dt
		if abs(missile_q) > missile_q_max:
			missile_q = np.sign(missile_q) * missile_q_max
		# (铅锤方向)由俯仰角速度计算俯仰角
		missile_theta = missile_theta_prev + missile_q * dt
		# (铅锤方向)俯仰角限幅
		missile_theta_max = np.deg2rad(45)  # 俯仰角限幅 [rad]
		if abs(missile_theta) > missile_theta_max:
			missile_theta = np.sign(missile_theta) * missile_theta_max

		# (水平方向)由偏航角加速度计算偏航角速度
		missile_r = missile_r_prev + missile_r_dot * dt
		if abs(missile_r) > missile_r_max:
			missile_r = np.sign(missile_r) * missile_r_max
		# (水平方向)由偏航角速度计算偏航角
		missile_psi = missile_psi_prev + missile_r * dt

		# 更新当前弹道倾角和弹道偏角
		missile_gamma = np.arctan2(missile_vel[1], missile_vel[0])
		missile_psis = np.arctan2(-missile_vel[2], missile_vel[0])

		# 计算迎角和侧滑角
		missile_alpha = missile_theta - missile_gamma
		missile_beta = missile_psi - missile_psis

		# 对侧滑角限幅(避免水平方向气动分离)
		if abs(missile_beta) > np.deg2rad(10):
			missile_psi = missile_psi_prev
			missile_beta = missile_psi - missile_psis

		# 由迎角和侧滑角计算气动力(升-阻-侧)
		if V_m > 1e-3:
			C_L = C_L_alpha * missile_alpha
			C_D = C_D_0 + K_ind * C_L * C_L
			C_C = C_C_beta * missile_beta
			lift_force = 0.5 * rho * V_m * V_m * S_ref * C_L
			drag_force = 0.5 * rho * V_m * V_m * S_ref * C_D
			side_force = 0.5 * rho * V_m * V_m * S_ref * C_C  # 注意：beta为+，侧力为-
		else:
			lift_force = 0.0
			side_force = 0.0
			drag_force = 0.0

		# 由气动力计算轴向力/法向力(上)/法向力(右)
		F_b_aero_x = - drag_force * np.cos(missile_alpha) * np.cos(missile_beta) \
					 + lift_force * np.sin(missile_alpha) \
					 - side_force * np.cos(missile_alpha) * np.sin(missile_beta)
		F_b_aero_y = + drag_force * np.sin(missile_alpha) * np.cos(missile_beta) \
					 + lift_force * np.cos(missile_alpha) \
					 + side_force * np.sin(missile_alpha) * np.sin(missile_beta)
		F_b_aero_z = - drag_force * np.sin(missile_beta) \
					 + lift_force * 0 \
					 + side_force * np.cos(missile_beta)

		# 计算弹体系气动力+推力的合力(特定力)
		F_b_ext_x = F_b_aero_x + missile_engine_thrust_t
		F_b_ext_y = F_b_aero_y
		F_b_ext_z = F_b_aero_z

		# 由轴向力和法向力，计算法向过载和法向过载(过载就是不考虑重力加速度的，当前版本只有气动力)
		n_X = F_b_ext_x / (missile_mass_t * g)
		n_Y = F_b_ext_y / (missile_mass_t * g)
		n_Z = F_b_ext_z / (missile_mass_t * g)

		# 由弹体系受力计算导航系受力(仅考虑气动力，为了检查是否有误差）
		F_n_aero_x = + F_b_aero_x * np.cos(missile_theta) * np.cos(missile_psi) \
					 - F_b_aero_y * np.sin(missile_theta) * np.cos(missile_psi) \
					 + F_b_aero_z * np.sin(missile_psi)
		F_n_aero_y = + F_b_aero_x * np.sin(missile_theta) \
					 + F_b_aero_y * np.cos(missile_theta) \
					 + F_b_aero_z * 0
		F_n_aero_z = - F_b_aero_x * np.cos(missile_theta) * np.sin(missile_psi) \
					 + F_b_aero_y * np.sin(missile_theta) * np.sin(missile_psi) \
					 + F_b_aero_z * np.cos(missile_psi)

		# 检查矩阵是否正确
		F_b_aero = np.linalg.norm([F_b_aero_x, F_b_aero_y, F_b_aero_z])
		F_n_aero = np.linalg.norm([F_n_aero_x, F_n_aero_y, F_n_aero_z])
		F_aero_dif = F_b_aero - F_n_aero

		# 由弹体系受力计算导航系受力(气动力+重力+推力）
		F_n_ext_x = + F_b_ext_x * np.cos(missile_theta) * np.cos(missile_psi) \
					- F_b_ext_y * np.sin(missile_theta) * np.cos(missile_psi) \
					+ F_b_ext_z * np.sin(missile_psi)
		F_n_ext_y = + F_b_ext_x * np.sin(missile_theta) \
					+ F_b_ext_y * np.cos(missile_theta) \
					+ F_b_ext_z * 0
		F_n_ext_z = - F_b_ext_x * np.cos(missile_theta) * np.sin(missile_psi) \
					+ F_b_ext_y * np.sin(missile_theta) * np.sin(missile_psi) \
					+ F_b_ext_z * np.cos(missile_psi)

		# 检查矩阵是否正确
		F_b_aero_vector = np.array([F_b_aero_x, F_b_aero_y, F_b_aero_z])
		F_b_ext_vector = np.array([F_b_ext_x, F_b_ext_y, F_b_ext_z])
		F_thrust_vector = F_b_ext_vector - F_b_aero_vector
		F_thrust_dif = np.linalg.norm(F_thrust_vector)

		# 在N系中合成总外力(气动力+重力+推力)
		F_n_total_x = F_n_ext_x + 0
		F_n_total_y = F_n_ext_y - missile_mass_t * g
		F_n_total_z = F_n_ext_z + 0

		# 由导航系受力，反算弹体系受力
		F_b_total_x = + F_n_total_x * np.cos(missile_theta) * np.cos(missile_psi) \
					  + F_n_total_y * np.sin(missile_theta) \
					  - F_n_total_z * np.cos(missile_theta) * np.sin(missile_psi)
		F_b_total_y = - F_n_total_x * np.sin(missile_theta) * np.cos(missile_psi) \
					  + F_n_total_y * np.cos(missile_theta) \
					  + F_n_total_z * np.sin(missile_theta) * np.sin(missile_psi)
		F_b_total_z = + F_n_total_x * np.sin(missile_psi) \
					  + F_n_total_y * 0 \
					  + F_n_total_z * np.cos(missile_psi)

		# 由弹体系受力，计算弹体系加速度
		acc_b_x = F_b_total_x / missile_mass_t
		acc_b_y = F_b_total_y / missile_mass_t
		acc_b_z = F_b_total_z / missile_mass_t  # 【负号加在了这一行】
		acc_b_V = [acc_b_x, acc_b_y, acc_b_z]
		acc_b = np.sqrt(acc_b_x * acc_b_x + acc_b_y * acc_b_y + acc_b_z * acc_b_z)

		# 计算导航系的加速度
		acc_n_x = F_n_total_x / missile_mass_t
		acc_n_y = F_n_total_y / missile_mass_t
		acc_n_z = F_n_total_z / missile_mass_t
		acc_n_V = [acc_n_x, acc_n_y, acc_n_z]
		acc_n = np.sqrt(acc_n_x * acc_n_x + acc_n_y * acc_n_y + acc_n_z * acc_n_z)

		# 对加速度积分，获取速度
		vel_n_x = vel_n_x_prev + acc_n_x * dt
		vel_n_y = vel_n_y_prev + acc_n_y * dt
		vel_n_z = vel_n_z_prev + acc_n_z * dt

		# 计算弹道系加速度
		V_t = np.linalg.norm([vel_n_x, vel_n_y, vel_n_z])
		missile_gamma_t = np.arctan2(vel_n_y, vel_n_x)
		missile_psis_t = np.arctan2(-vel_n_z, vel_n_x)
		acc_2_x = (V_t - V_m) / dt
		acc_2_y = V_t * (missile_gamma_t - missile_gamma) / dt
		acc_2_z = -V_t * np.cos(missile_gamma_t) * (missile_psis_t - missile_psis) / dt
		acc_2_V = [acc_2_x, acc_2_y, acc_2_z]
		acc_2 = np.sqrt(acc_2_x * acc_2_x + acc_2_y * acc_2_y + acc_2_z * acc_2_z)

		# 计算由弹体系指令加速度计算弹道系指令加速度
		C_tb = [
			[np.cos(missile_alpha) * np.cos(missile_beta), -np.sin(missile_alpha) * np.cos(missile_beta), np.sin(missile_beta)],
			[np.sin(missile_alpha), np.cos(missile_beta), 0],
			[-np.cos(missile_alpha) * np.sin(missile_beta), np.sin(missile_alpha) * np.sin(missile_beta), np.cos(missile_beta)]
		]
		# 使用 @ 运算符计算弹道系指令加速度
		acc_cmd = np.array([0, acc_cmd_v, acc_cmd_h])
		acc_2C_x, acc_2C_v, acc_2C_h = C_tb @ acc_cmd  # 简洁直观！

		# 计算加速度误差
		ac_b = np.linalg.norm([acc_cmd_v, acc_cmd_h])
		ac_2 = np.linalg.norm([acc_2C_x, acc_2C_v, acc_2C_h])
		ac_2_vh = np.linalg.norm([acc_2C_v, acc_2C_h])
		ac_dif = ac_b - ac_2
		ac_dif_vh = ac_b - ac_2_vh

		# 对速度积分，获取位置(G系)
		pos_g_x = pos_g_x_prev + vel_n_x * dt
		pos_g_y = pos_g_y_prev + vel_n_y * dt
		pos_g_z = pos_g_z_prev + vel_n_z * dt

		# 更新上一时刻舵偏角速率
		delta_e_dot_prev = delta_e_dot
		# 更新上一时刻舵偏角
		delta_e_prev = delta_e
		delta_r_prev = delta_r
		# 更新上一时刻俯仰角速度
		missile_q_prev = missile_q
		missile_r_prev = missile_r
		# 更新上一时刻俯仰角
		missile_theta_prev = missile_theta
		missile_psi_prev = missile_psi
		# 更新上一时刻的速度
		missile_vel[0] = vel_n_x
		missile_vel[1] = vel_n_y
		missile_vel[2] = vel_n_z
		vel_n_x_prev = vel_n_x
		vel_n_y_prev = vel_n_y
		vel_n_z_prev = vel_n_z
		# 更新导弹位置
		missile_pos[0] = pos_g_x
		missile_pos[1] = pos_g_y
		missile_pos[2] = pos_g_z
		pos_g_x_prev = pos_g_x
		pos_g_y_prev = pos_g_y
		pos_g_z_prev = pos_g_z

		# 存储数据（顺序：舵面->姿态->速度->位置，T在前M在后）
		target_pos_history[i] = target_pos
		err_h_history[i] = err_h
		delta_e_history[i] = np.rad2deg(delta_e)
		delta_r_history[i] = np.rad2deg(delta_r)
		missile_mass_history[i] = missile_mass_t
		missile_thrust_history[i] = missile_engine_thrust_t
		missile_theta_history[i] = np.rad2deg(missile_theta)
		missile_gamma_history[i] = np.rad2deg(missile_gamma)
		missile_alpha_history[i] = np.rad2deg(missile_alpha)
		missile_psi_history[i] = np.rad2deg(missile_psi)
		missile_psis_history[i] = np.rad2deg(missile_psis)
		missile_beta_history[i] = np.rad2deg(missile_beta)
		missile_lift_history[i] = lift_force
		missile_drag_history[i] = drag_force
		missile_side_history[i] = side_force
		missile_acc1s_history[i] = acc_b
		missile_acc1v_history[i] = acc_b_V
		missile_acc2s_history[i] = acc_2
		missile_acc2v_history[i] = acc_2_V
		missile_acc2C_history[i] = [acc_2C_v, acc_2C_h]
		missile_accNs_history[i] = acc_n
		missile_accNv_history[i] = acc_n_V
		missile_speed_history[i] = V_m
		missile_vel_history[i] = missile_vel
		missile_pos_history[i] = missile_pos
		missile_nx_history[i] = n_X
		missile_ny_history[i] = n_Y
		missile_nz_history[i] = n_Z
		missile_lambda_history[i] = [np.rad2deg(lambda_v), np.rad2deg(lambda_h)]
		missile_accC_history[i] = [acc_cmd_v, acc_cmd_h]

		# 打印
		if i > 0 and plt_yes:
			print(f'---{i}, m_pos=[{missile_pos[0]:.2f}, {missile_pos[1]:.2f}, {missile_pos[2]:.2f}], '
				  f'm_gamma={missile_gamma * 57.3:.2f}, '
				  f'mt_d={mt_dist:.2f}, '
				  f'delta_e={delta_e * 57.3:.2f}, ac_cmd_v={acc_cmd_v:.2f}, err_v={err_v:.2f}, '
				  f'F_aero_dif={F_aero_dif:.2f}, F_thrust_dif={F_thrust_dif:.2f}, '
				  f'ac_dif={ac_dif:.2f}, ac_dif_vh={ac_dif_vh:.2f}, '
				  f'delta_r={delta_r * 57.3:.2f}, ac_cmd_h={acc_cmd_h:.2f}, err_h={err_h:.2f}, '
				  f'C={side_force:.2f}, beta={missile_beta * 57.3:.2f}, psis={missile_psis * 57.3:.2f}')

	# 绘制结果
	plt_num_h = 6
	plt_num_v = 4
	plt.figure(figsize=(24, 9))
	t = np.arange(stop_i) * dt
	plt_i = 0

	# 导弹姿态角
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_theta_history[1:len(t)], label='俯仰角 θ [deg]')
	plt.plot(t[1:], missile_alpha_history[1:len(t)], label='迎角 α [deg]')
	plt.plot(t[1:], missile_gamma_history[1:len(t)], label='弹道倾角 γ [deg]')
	plt.xlabel('时间 [s]')
	plt.ylabel('角度 [deg]')
	plt.title('导弹姿态角（铅锤，θ=α+γ）')
	plt.legend()
	plt.grid(True)

	# 导弹姿态角
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_psi_history[1:len(t)], label='偏航角 ψ [deg]')
	plt.plot(t[1:], missile_beta_history[1:len(t)], label='侧滑角 β [deg]')
	plt.plot(t[1:], missile_psis_history[1:len(t)], label='弹道偏角 ψs [deg]')
	plt.xlabel('时间 [s]')
	plt.ylabel('角度 [deg]')
	plt.title('导弹姿态角（水平，ψ=β+ψs）')
	plt.legend()
	plt.grid(True)

	# 导弹舵偏角
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], delta_r_history[1:len(t)], label='舵偏角 δr [deg]')
	plt.xlabel('时间 [s]')
	plt.ylabel('角度 [deg]')
	plt.title('导弹舵偏角（δr）')
	plt.legend()
	plt.grid(True)

	# 导弹加速度
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_acc1v_history[:, 0][1:len(t)], label='ab_x')
	plt.plot(t[1:], missile_acc1v_history[:, 1][1:len(t)], label='ab_y')
	plt.plot(t[1:], missile_acc1v_history[:, 2][1:len(t)], label='ab_z')
	plt.plot(t[1:-plt_stop], missile_accC_history[:, 0][1:len(t) - plt_stop], label='ac_V')
	plt.plot(t[1:-plt_stop], missile_accC_history[:, 1][1:len(t) - plt_stop], label='ac_H')
	plt.xlabel('时间 [s]')
	plt.ylabel('加速度 [m/s^2]')
	plt.title('导弹加速度（弹体系）')
	plt.legend()
	plt.grid(True)

	# 导弹加速度
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_acc2v_history[:, 0][1:len(t)], label='a2_x')
	plt.plot(t[1:], missile_acc2v_history[:, 1][1:len(t)], label='a2_y')
	plt.plot(t[1:], missile_acc2v_history[:, 2][1:len(t)], label='a2_z')
	plt.plot(t[1:-plt_stop], missile_acc2C_history[:, 0][1:len(t) - plt_stop], label='a2c_V')
	plt.plot(t[1:-plt_stop], missile_acc2C_history[:, 1][1:len(t) - plt_stop], label='a2c_H')
	plt.xlabel('时间 [s]')
	plt.ylabel('加速度 [m/s^2]')
	plt.title('导弹加速度（弹道系）')
	plt.legend()
	plt.grid(True)

	# 导弹速度
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_speed_history[1:len(t)], label='速度 V [m/s]')
	plt.xlabel('时间 [s]')
	plt.ylabel('速度 [m/s]')
	plt.title('导弹速度（V）')
	plt.legend()
	plt.grid(True)

	# 导弹舵偏角
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], delta_e_history[1:len(t)], label='舵偏角 δe [deg]')
	plt.xlabel('时间 [s]')
	plt.ylabel('角度 [deg]')
	plt.title('导弹舵偏角（δe）')
	plt.legend()
	plt.grid(True)

	# 导弹升力
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_lift_history[1:len(t)], label='升力 L [N]')
	plt.xlabel('时间 [s]')
	plt.ylabel('升力 [N]')
	plt.title('导弹升力（L）')
	plt.legend()
	plt.grid(True)

	# 导弹阻力
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_drag_history[1:len(t)], label='阻力 D [N]')
	plt.xlabel('时间 [s]')
	plt.ylabel('阻力 [N]')
	plt.title('导弹阻力（D）')
	plt.legend()
	plt.grid(True)

	# 导弹侧力
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_side_history[1:len(t)], label='侧力 C [N]')
	plt.xlabel('时间 [s]')
	plt.ylabel('侧力 [N]')
	plt.title('导弹侧力（C）')
	plt.legend()
	plt.grid(True)

	# 弹目距离
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], mt_dist_history[1:len(t)], label='弹目距离 mt_d [m]')
	plt.xlabel('时间 [s]')
	plt.ylabel('弹目距离 [m]')
	plt.title('弹目距离 （mt_d）')
	plt.legend()
	plt.grid(True)

	# 弹道
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(missile_pos_history[:, 0][1:len(t)], missile_pos_history[:, 1][1:len(t)], label='导弹轨迹')
	plt.plot(target_pos_history[:, 0][1:len(t)], target_pos_history[:, 1][1:len(t)], label='目标轨迹')
	plt.xlabel('X(北) 位置 [m]')
	plt.ylabel('Z(天) 位置 [m]')
	plt.title('导弹和目标轨迹（铅锤）')
	plt.legend()
	plt.grid(True)
	plt.axis('equal')  # 保证x和y轴比例相同，轨迹不变形

	# 弹道
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(missile_pos_history[:, 2][1:len(t)], missile_pos_history[:, 0][1:len(t)], label='导弹轨迹')
	plt.plot(target_pos_history[:, 2][1:len(t)], target_pos_history[:, 0][1:len(t)], label='目标轨迹')
	plt.xlabel('Y(东) 位置 [m]')
	plt.ylabel('X(北) 位置 [m]')
	plt.title('导弹和目标轨迹（水平）')
	plt.legend()
	plt.grid(True)
	plt.axis('equal')  # 保证x和y轴比例相同，轨迹不变形

	# 导弹速度
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_vel_history[:, 0][1:len(t)], label='Vx')
	plt.plot(t[1:], missile_vel_history[:, 1][1:len(t)], label='Vy')
	plt.plot(t[1:], missile_vel_history[:, 2][1:len(t)], label='Vz')
	plt.xlabel('时间 [s]')
	plt.ylabel('速度 [m/s]')
	plt.title('导弹速度（V）')
	plt.legend()
	plt.grid(True)

	# 过载（弹体系x）
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_nx_history[1:len(t)], label='弹体x过载')
	plt.xlabel('时间 [s]')
	plt.ylabel('x过载')
	plt.title('弹体x过载')
	plt.legend()
	plt.grid(True)

	# 过载（弹体系y）
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_ny_history[1:len(t)], label='弹体y过载')
	plt.xlabel('时间 [s]')
	plt.ylabel('y过载')
	plt.title('弹体y过载')
	plt.legend()
	plt.grid(True)

	# 过载（弹体系z）
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_nz_history[1:len(t)], label='弹体z过载')
	plt.xlabel('时间 [s]')
	plt.ylabel('z过载')
	plt.title('弹体z过载')
	plt.legend()
	plt.grid(True)

	# 水平误差
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:-plt_stop], err_h_history[1:len(t) - plt_stop], label='水平误差')
	plt.xlabel('时间 [s]')
	plt.ylabel('加速度 [m/s^2]')
	plt.title('水平误差')
	plt.legend()
	plt.grid(True)

	# 视线角
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_lambda_history[:, 0][1:len(t)], label='视线角v')
	plt.plot(t[1:], missile_lambda_history[:, 1][1:len(t)], label='视线角h')
	plt.xlabel('时间 [s]')
	plt.ylabel('视线角 [deg]')
	plt.title('视线角')
	plt.legend()
	plt.grid(True)

	# 导弹质量
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_mass_history[1:len(t)], label='导弹质量')
	plt.xlabel('时间 [s]')
	plt.ylabel('导弹质量 [kg]')
	plt.title('导弹质量')
	plt.legend()
	plt.grid(True)

	# 导弹推力
	plt_i = plt_i + 1
	plt.subplot(plt_num_v, plt_num_h, plt_i)
	plt.plot(t[1:], missile_thrust_history[1:len(t)], label='导弹推力')
	plt.xlabel('时间 [s]')
	plt.ylabel('导弹推力 [N]')
	plt.title('导弹推力')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()

	print(f'missile_gamma_max={missile_gamma_max * 57.3:.2f}')
	ratio_control_yes = gamma_control_yes / stop_i
	print(f'ratio_control_yes={ratio_control_yes*100:.2f}%')

	if plt_yes:
		plt.show(block=False)  # block=False 允许非阻塞显示
		plt.pause(300)  # 暂停2秒
		# plt.savefig(f'./gam/{int(ratio_control_yes * 100)}_{KPi + 1}_{KIi + 1}_{KDi + 1}.png')
	plt.close()  # 关闭当前图像

	return fix_v, fix_h


if __name__ == "__main__":
	# FIX_V, FIX_H = proportional_guidance()
	FIX_V = FIX_H = 0.01

	proportional_guidance(fix_v=FIX_V, fix_h=FIX_H)

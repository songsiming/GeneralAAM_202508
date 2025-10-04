"""
1） 实现基本可视化功能
2) 添加捕获视角的代码
3) 实时输出视角
"""
import time

import numpy as np
from mayavi import mlab
from pyface.timer.api import Timer


def print_current_view():
	"""打印当前视角"""
	azimuth, elevation, distance, focalpoint = mlab.view()
	_ = mlab.gcf().scene.camera

	print(f"视角: azimuth={azimuth:6.1f}°, elevation={elevation:6.1f}°, distance={distance:6.1f}, "
		  f"焦点=({focalpoint[0]:.1f}, {focalpoint[1]:.1f}, {focalpoint[2]:.1f})")


class ViewTracker:
	def __init__(self, update_interval=1.0):  # 每秒更新一次
		self.last_view = None
		self.update_interval = update_interval
		self.last_update = time.time()
		self.timer = None  # 添加timer引用

	def start_tracking(self):
		"""启动视角跟踪"""

		def update_callback():
			self.check_view_change()

		# 启动定时器（每100毫秒检查一次）
		self.timer = Timer(100, update_callback)
		print("视角跟踪已启动...")

	def check_view_change(self):
		"""检查视角是否变化并输出"""
		current_time = time.time()
		if current_time - self.last_update < self.update_interval:
			return

		current_view = mlab.view()
		if self.last_view is None or not np.allclose(current_view[:3], self.last_view[:3], rtol=0.01):
			self.last_view = current_view
			self.last_update = current_time
			print_current_view()


def create_missile_3d_mayavi(tracker):
	# 导弹参数
	missile_nose_angle = 15  # 导弹弹头锥角 [deg]
	missile_body_d = 0.178  # 导弹直径 [m]
	missile_load_l = 1.65  # 导弹载荷段长度(决定总长度) [m]
	missile_engine_l = 2.00  # 导弹动力段长度(决定总长度) [m]
	missile_nozzle_l = 0.1  # 导弹尾喷管长度 [m]
	missile_wing_surf = 0.04  # 弹翼面积

	# 计算半径
	r = missile_body_d / 2

	# 弹头锥角30度
	nose_angle = np.radians(missile_nose_angle)
	nose_length = r / np.tan(nose_angle)  # 弹头长度

	# 创建新场景
	mlab.figure(size=(1200, 900), bgcolor=(1, 1, 1))

	# 1. 创建弹头锥体(创建弹头锥体参数化坐标网格)
	x_nose = np.linspace(-nose_length, 0, 50)  # 从0到弹头长度的50个等间距点(纵向)
	theta = np.linspace(0, 2 * np.pi, 50)  # 从0到2π的50个等间距点(角度，相当于0°到360°)
	x_nose_grid, theta_grid = np.meshgrid(x_nose, theta)

	r_nose = r * (x_nose_grid / nose_length)
	y_nose = r_nose * np.sin(theta_grid)
	z_nose = r_nose * np.cos(theta_grid)

	mlab.mesh(x_nose_grid, y_nose, z_nose, color=(0.6, 0.6, 0.6), opacity=0.9)

	# 2. 创建载荷段圆柱体
	x_load_start = -nose_length
	x_load_end = -missile_load_l
	x_load = np.linspace(x_load_start, x_load_end, 50)
	x_load_grid, theta_grid = np.meshgrid(x_load, theta)

	y_load = r * np.sin(theta_grid)
	z_load = r * np.cos(theta_grid)

	mlab.mesh(x_load_grid, y_load, z_load, color=(0.4, 0.6, 0.8), opacity=0.9, name='Payload Section')

	# 3. 创建动力段圆柱体
	x_engine_start = x_load_end
	x_engine_end = x_engine_start - (missile_engine_l - missile_nozzle_l)
	x_engine = np.linspace(x_engine_start, x_engine_end, 50)
	x_engine_grid, theta_grid = np.meshgrid(x_engine, theta)

	y_engine = r * np.sin(theta_grid)
	z_engine = r * np.cos(theta_grid)

	mlab.mesh(x_engine_grid, y_engine, z_engine, color=(0.2, 0.3, 0.6), opacity=0.9, name='Engine Section')

	# 创建弹翼（直角三角形，长边贴着弹体）
	wing_chord = np.sqrt(missile_wing_surf * 2 / 3)  # 翼弦长（短边）
	wing_span = wing_chord * 3  # 翼展（长边）

	# 弹翼位置（动力段尾部）
	wing_x_end = x_engine_end + 0.02  # 弹尾减去2cm(反方向用加)
	wing_length = wing_span  # 长边长度
	wing_width = wing_chord  # 短边宽度

	# 基础顶点（在ZX平面，Y=0）
	vertices_base = np.array([
		[wing_x_end, 0, r],  # 顶点0: 贴在弹体上
		[wing_x_end + wing_length, 0, r],  # 顶点1: 轴向延伸（长边）
		[wing_x_end, 0, r + wing_width]  # 顶点2: 径向延伸（短边）
	])

	# 4个弹翼的安装角度
	angles = [0, 90, 180, 270]
	wing_colors = [(1, 0, 0), (0, 1, 0), (0.8, 0.8, 0), (0, 1, 1)]

	# 循环4个弹翼
	for i, angle in enumerate(angles):
		# 旋转矩阵（绕X轴旋转）
		rad_angle = np.radians(angle)
		rot_matrix = np.array([
			[1, 0, 0],
			[0, np.cos(rad_angle), -np.sin(rad_angle), ],
			[0, np.sin(rad_angle), np.cos(rad_angle)]
		])

		# 旋转顶点到正确位置
		vertices_rotated = np.dot(vertices_base, rot_matrix.T)

		# 三角形面(定义三角形顶点连接方式)
		triangles = [(0, 1, 2)]

		# 绘制翼面
		mlab.triangular_mesh(vertices_rotated[:, 0], vertices_rotated[:, 1], vertices_rotated[:, 2], triangles, color=wing_colors[i % 4], opacity=0.8)

		# 可选：添加翼面边框以便更好观察
		mlab.plot3d(vertices_rotated[[0, 1], 0],  # 两个顶点的x坐标
					vertices_rotated[[0, 1], 1],  # 两个顶点的y坐标
					vertices_rotated[[0, 1], 2],  # 两个顶点的z坐标
					color=wing_colors[i % 4], tube_radius=0.005)
		mlab.plot3d(vertices_rotated[[0, 2], 0],  # 顶点0和2的x坐标
					vertices_rotated[[0, 2], 1],  # 顶点0和2的y坐标
					vertices_rotated[[0, 2], 2],  # 顶点0和2的z坐标
					color=wing_colors[i % 4], tube_radius=0.005)
		mlab.plot3d(vertices_rotated[[1, 2], 0],  # 顶点1和2的x坐标
					vertices_rotated[[1, 2], 1],  # 顶点1和2的y坐标
					vertices_rotated[[1, 2], 2],  # 顶点1和2的z坐标
					color=wing_colors[i % 4], tube_radius=0.005)

	# 5. 添加尾喷管
	x_nozzle_start = x_engine_end
	x_nozzle_end = x_nozzle_start - missile_nozzle_l
	x_nozzle = np.linspace(x_nozzle_start, x_nozzle_end, 20)
	x_nozzle_grid, theta_grid = np.meshgrid(x_nozzle, theta)

	# 喷管略微扩张
	r_nozzle = r * (1 - 0.1 * (x_nozzle_grid - x_nozzle_start) / missile_nozzle_l)
	y_nozzle = r_nozzle * np.sin(theta_grid)
	z_nozzle = r_nozzle * np.cos(theta_grid)

	mlab.mesh(x_nozzle_grid, y_nozzle, z_nozzle, color=(0.3, 0.3, 0.3), opacity=0.7, name='Nozzle')

	# 计算总长
	total_length = missile_load_l + missile_engine_l

	# 设置视图
	mlab.view(azimuth=160, elevation=40, distance=6, focalpoint=(-total_length / 1.8, 0, 0))
	# 获取相机并设置"上"方向为Y轴
	camera = mlab.gcf().scene.camera
	camera.view_up = [0, 1, 0]  # Y轴朝上

	# 先添加坐标轴，然后单独调整Z轴
	axes = mlab.axes(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]', color=(1, 0, 0))

	# 手动设置范围，让Z轴有足够的空间显示刻度
	axes.axes.bounds = (-total_length, 0, -0.3, 0.3, -0.3, 0.3)

	# 设置刻度属性
	axes.axes.number_of_labels = 5
	axes.axes.label_format = "%.1f"  # 标签格式
	axes.axes.font_factor = 1.2  # 字体大小因子
	axes.axes.axis_label_text_property.color = (0, 0, 0)  # 黑色标签
	axes.axes.axis_title_text_property.color = (0, 0, 0)  # 黑色标题

	# 添加标题
	title = mlab.title(f'AIM-120 Missile\nMissile L: {total_length:.2f}m, Missile D: {missile_body_d:.3f}m', size=0.3, color=(0, 0, 0))
	# 获取标题的TextProperty并设置字体
	title.actor.text_property.font_family = 'times'  # 字体家族
	title.actor.text_property.font_size = 24  # 字体大小
	title.actor.text_property.color = (0, 0, 0)  # 颜色
	title.actor.text_property.bold = True  # 粗体
	title.actor.text_property.italic = False  # 斜体

	# 添加图例文本
	text_y_pos = 0.4
	text_z_pos = 0.2
	mlab.text3d(-total_length * 0.2, text_y_pos, text_z_pos, f"Warhead Angle({missile_nose_angle}')", scale=0.05, color=(0.6, 0.6, 0.6))
	mlab.text3d(-total_length * 0.4, text_y_pos, text_z_pos, 'Load Section', scale=0.05, color=(0.4, 0.6, 0.8))
	mlab.text3d(-total_length * 0.8, text_y_pos, text_z_pos, 'Engine Section', scale=0.05, color=(0.2, 0.3, 0.6))
	mlab.text3d(-total_length * 1.0, text_y_pos, text_z_pos, 'Wing', scale=0.05, color=(1, 0, 0))

	# 打印参数信息
	print("=" * 50)
	print("导弹三维模型参数:")
	print("=" * 50)
	print(f"总长度:\t\t{total_length:.3f} m")
	print(f"弹头长度:\t{nose_length:.3f} m ({missile_nose_angle}°锥角)")
	print(f"载荷段长度:\t{missile_load_l:.3f} m")
	print(f"动力段长度:\t{missile_engine_l:.3f} m")
	print(f"尾喷口长度:\t{missile_nozzle_l:.3f} m")
	print(f"直径:\t\t{missile_body_d:.3f} m")
	print(f"弹翼面积:\t{missile_wing_surf:.3f} m²")
	print(f"弹翼展长:\t{wing_span:.3f} m")
	print(f"弹翼弦长:\t{wing_chord:.3f} m")
	print("=" * 50)

	# 启动视角跟踪
	tracker.start_tracking()

	print("开始旋转模型，视角参数将自动输出...")
	print("旋转模型后，视角变化会自动显示在控制台")

	# 显示图形
	mlab.show()


# 运行可视化
if __name__ == "__main__":
	# 创建跟踪器
	Tracker = ViewTracker()

	create_missile_3d_mayavi(Tracker)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore')


# ==================== 1. 读取CSV文件 ====================
def read_thrust_data(file_path):
	"""
	读取包含推力数据的CSV文件

	参数:
		file_path: CSV文件路径

	返回:
		DataFrame: 包含时间和推力数据的DataFrame
	"""
	try:
		# 尝试读取CSV文件
		df = pd.read_csv(file_path,encoding='gbk')
		print("CSV文件读取成功!")
		print(f"数据形状: {df.shape}")
		print("\n数据前5行:")
		print(df.head())
		print("\n数据列名:")
		print(df.columns.tolist())

		return df
	except FileNotFoundError:
		print(f"错误: 找不到文件 '{file_path}'")
		return None
	except Exception as e:
		print(f"读取文件时发生错误: {str(e)}")
		return None


# ==================== 2. 数据预处理 ====================
def preprocess_data(df):
	"""
	预处理数据，确保时间列和推力列存在且格式正确

	参数:
		df: 原始DataFrame

	返回:
		tuple: (时间数据, 推力数据, 清理后的列名)
	"""
	# 尝试不同的时间列名
	time_columns = ['时间 Time(s)', 'Time(s)', 'Time', 'time', 't', '时间']
	thrust_columns = ['推力 m_thrust(N)', 'm_thrust(N)', '推力', 'thrust', 'F', 'Thrust']

	time_col = None
	thrust_col = None

	# 查找时间列
	for col in time_columns:
		if col in df.columns:
			time_col = col
			break

	# 查找推力列
	for col in thrust_columns:
		if col in df.columns:
			thrust_col = col
			break

	if time_col is None:
		# 尝试通过索引或数据类型猜测时间列
		numeric_cols = df.select_dtypes(include=[np.number]).columns
		if len(numeric_cols) >= 1:
			time_col = numeric_cols[0]
			print(f"警告: 未找到标准时间列名，使用 '{time_col}' 作为时间")
		else:
			print("错误: 未找到时间列")
			return None, None, None, None

	if thrust_col is None:
		# 尝试通过列名包含"推力"或"thrust"来猜测
		for col in df.columns:
			if '推力' in col or 'thrust' in col.lower():
				thrust_col = col
				break

		if thrust_col is None and len(df.columns) >= 2:
			# 使用第二列作为推力列
			thrust_col = df.columns[1] if time_col == df.columns[0] else df.columns[0]
			print(f"警告: 未找到标准推力列名，使用 '{thrust_col}' 作为推力")
		else:
			print("错误: 未找到推力列")
			return None, None, None, None

	# 提取数据
	time_data = df[time_col].values
	thrust_data = df[thrust_col].values

	# 清理数据：移除NaN值
	valid_mask = ~np.isnan(time_data) & ~np.isnan(thrust_data)
	time_data = time_data[valid_mask]
	thrust_data = thrust_data[valid_mask]

	print(f"使用的时间列: '{time_col}'")
	print(f"使用的推力列: '{thrust_col}'")
	print(f"有效数据点数: {len(time_data)}")

	return time_data, thrust_data, time_col, thrust_col


# ==================== 3. 创建美观的图表 ====================
def create_beautiful_thrust_plot(time_data, thrust_data, time_col, thrust_col, save_path=None):
	"""
	创建美观的推力-时间趋势图

	参数:
		time_data: 时间数据数组
		thrust_data: 推力数据数组
		time_col: 时间列名
		thrust_col: 推力列名
		save_path: 保存图片的路径（可选）
	"""
	# 设置中文字体（如果系统中有中文字体）
	try:
		# 尝试使用系统中文字体
		font_paths = [
			'C:/Windows/Fonts/simhei.ttf',  # Windows
			'/System/Library/Fonts/STHeiti Medium.ttc',  # macOS
			'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux
		]

		for font_path in font_paths:
			try:
				fm.fontManager.addfont(font_path)
				prop = fm.FontProperties(fname=font_path)
				plt.rcParams['font.sans-serif'] =  [prop.get_name()] # [prop.get_name()]  ['SimSun', 'Times New Roman']
				plt.rcParams['axes.unicode_minus'] = False
				print(f"使用中文字体: {prop.get_name()}")
				break
			except:
				continue
	except:
		print("使用默认英文字体")

	# 创建图形和坐标轴
	fig, ax = plt.subplots(figsize=(14, 8))

	# 设置图形背景颜色
	fig.patch.set_facecolor('#f8f9fa')
	ax.set_facecolor('#ffffff')

	# 绘制推力-时间曲线
	# 使用渐变色填充曲线下方区域
	line = ax.plot(time_data, thrust_data,
				   color='#2E86AB',  # 主色调：深蓝色
				   linewidth=2.5,
				   marker='o',
				   markersize=4,
				   markerfacecolor='#FFFFFF',
				   markeredgecolor='#2E86AB',
				   markeredgewidth=1.5,
				   label='推力变化趋势',
				   zorder=3)

	# 填充曲线下方区域
	ax.fill_between(time_data, thrust_data,
					alpha=0.15,
					color='#2E86AB',
					zorder=2)

	# 添加关键数据点标记（最大值、最小值、阶段变化点）
	max_thrust_idx = np.argmax(thrust_data)
	min_thrust_idx = np.argmin(thrust_data)

	# 标记最大推力点
	ax.scatter(time_data[max_thrust_idx], thrust_data[max_thrust_idx],
			   color='#D64933', s=150, zorder=4,
			   edgecolors='#FFFFFF', linewidth=2,
			   label=f'最大推力: {thrust_data[max_thrust_idx]:.0f} N')

	# 标记最小推力点
	ax.scatter(time_data[min_thrust_idx], thrust_data[min_thrust_idx],
			   color='#00A878', s=150, zorder=4,
			   edgecolors='#FFFFFF', linewidth=2,
			   label=f'最小推力: {thrust_data[min_thrust_idx]:.0f} N')

	# 添加文本标注
	ax.annotate(f'{thrust_data[max_thrust_idx]:.0f} N',
				xy=(time_data[max_thrust_idx], thrust_data[max_thrust_idx]),
				xytext=(time_data[max_thrust_idx], thrust_data[max_thrust_idx] * 1.05),
				fontsize=11, fontweight='bold', color='#D64933',
				ha='center', va='bottom',
				arrowprops=dict(arrowstyle='->', color='#D64933', lw=1.5))

	# 添加阶段划分线（如果有明显的阶段变化）
	# 这里我们假设推力数据可能有阶段性变化，自动检测
	# if len(thrust_data) > 10:
	# 	# 计算推力变化的梯度
	# 	thrust_gradient = np.gradient(thrust_data)
	# 	# 找出梯度变化较大的点
	# 	gradient_abs = np.abs(thrust_gradient)
	# 	threshold = np.percentile(gradient_abs, 90)  # 取梯度绝对值最大的前10%
	# 	phase_change_indices = np.where(gradient_abs > threshold)[0]
	#
	# 	# 添加阶段分割线
	# 	for idx in phase_change_indices:
	# 		if 0 < idx < len(time_data) - 1:
	# 			ax.axvline(x=time_data[idx], color='#A23B72',
	# 					   linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

	# 设置坐标轴标签
	ax.set_xlabel(time_col, fontsize=14, fontweight='bold', color='#333333', labelpad=10)
	ax.set_ylabel(thrust_col, fontsize=14, fontweight='bold', color='#333333', labelpad=10)

	# 设置标题
	ax.set_title('单室双推发动机推力随时间变化趋势',
				 fontsize=18, fontweight='bold', color='#2E3A59', pad=20)

	# 设置坐标轴范围
	ax.set_xlim([time_data[0] - 0.02 * (time_data[-1] - time_data[0]),
				 time_data[-1] + 0.02 * (time_data[-1] - time_data[0])])

	# 确保y轴从0或最小值开始
	y_min = min(0, np.min(thrust_data) * 1.1) if np.min(thrust_data) > 0 else np.min(thrust_data) * 1.1
	y_max = np.max(thrust_data) * 1.1
	ax.set_ylim([y_min, y_max])

	# 设置坐标轴刻度和网格
	ax.xaxis.set_major_locator(MultipleLocator((time_data[-1] - time_data[0]) / 10))
	ax.xaxis.set_minor_locator(AutoMinorLocator(5))
	ax.yaxis.set_major_locator(MultipleLocator((y_max - y_min) / 10))
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))

	# 设置网格线
	ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7, color='#dddddd')
	ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5, color='#eeeeee')

	# 设置坐标轴边框
	for spine in ax.spines.values():
		spine.set_linewidth(1.5)
		spine.set_color('#444444')

	# 添加图例
	ax.legend(loc='upper right', fontsize=12, framealpha=0.95,
			  edgecolor='#333333', facecolor='#f8f9fa')

	# 添加统计信息文本框
	stats_text = f"""
统计摘要:
• 数据点数: {len(time_data):,}
• 时间范围: {time_data[0]:.2f} - {time_data[-1]:.2f} s
• 推力范围: {np.min(thrust_data):.0f} - {np.max(thrust_data):.0f} N
• 平均推力: {np.mean(thrust_data):.0f} N
• 推力标准差: {np.std(thrust_data):.0f} N
• 总冲量: {np.trapz(thrust_data, time_data):,.0f} N·s
"""

	ax.text(0.72, 0.58, stats_text, transform=ax.transAxes,
			fontsize=10, verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='#f8f9fa',
					  alpha=0.9, edgecolor='#cccccc', pad=2))

	# 调整布局
	plt.tight_layout()

	# 保存图表（如果指定了保存路径）
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
		print(f"\n图表已保存到: {save_path}")

	# 显示图表
	plt.show()

	return fig, ax


# ==================== 4. 主函数 ====================
def main():
	"""
	主函数：执行数据读取、处理和可视化
	"""
	print("=" * 60)
	print("单室双推发动机推力数据可视化")
	print("=" * 60)

	# 设置CSV文件路径（请修改为您的实际文件路径）
	csv_file_path = "result_file.csv"  # 替换为您的CSV文件路径

	# 1. 读取数据
	df = read_thrust_data(csv_file_path)
	if df is None:
		return

	print("\n" + "=" * 60)
	print("数据预处理")
	print("=" * 60)

	# 2. 预处理数据
	time_data, thrust_data, time_col, thrust_col = preprocess_data(df)

	if time_data is None or thrust_data is None:
		print("数据预处理失败，请检查CSV文件格式。")
		return

	print("\n" + "=" * 60)
	print("生成可视化图表")
	print("=" * 60)

	# 3. 创建图表
	fig, ax = create_beautiful_thrust_plot(
		time_data, thrust_data, time_col, thrust_col,
		save_path="thrust_vs_time_plot.png"  # 可选的保存路径
	)

	print("\n" + "=" * 60)
	print("可视化完成！")
	print("=" * 60)


# ==================== 5. 生成示例数据（如果实际数据不可用） ====================
def generate_sample_data(file_path="thrust_data.csv"):
	"""
	生成示例推力数据（用于测试，如果实际数据不可用）

	参数:
		file_path: 保存示例数据的CSV文件路径
	"""
	print("生成示例数据...")

	# 模拟单室双推发动机的推力数据
	np.random.seed(42)  # 固定随机种子以便结果可重复

	# 时间数据：0到130秒，每0.1秒一个点
	time = np.arange(0, 130.1, 0.1)

	# 推力数据：模拟两个推力阶段
	thrust = np.zeros_like(time)

	for i, t in enumerate(time):
		if t < 30:  # 第一阶段：大推力
			thrust[i] = 5000 + np.random.normal(0, 100)  # 5000N ± 100N噪声
		elif t < 130:  # 第二阶段：小推力
			thrust[i] = 1000 + np.random.normal(0, 50)  # 1000N ± 50N噪声
		else:  # 发动机关闭
			thrust[i] = 0

	# 添加一些平滑过渡
	transition_start = 28
	transition_end = 32
	transition_mask = (time >= transition_start) & (time <= transition_end)
	if np.any(transition_mask):
		# 在过渡区域平滑推力变化
		transition_idx = np.where(transition_mask)[0]
		for idx in transition_idx:
			t_frac = (time[idx] - transition_start) / (transition_end - transition_start)
			thrust[idx] = 5000 * (1 - t_frac) + 1000 * t_frac + np.random.normal(0, 80)

	# 创建DataFrame
	df = pd.DataFrame({
		'时间 Time(s)': time,
		'推力 m_thrust(N)': thrust
	})

	# 保存到CSV
	df.to_csv(file_path, index=False, encoding='utf-8-sig')
	print(f"示例数据已保存到: {file_path}")
	print(f"数据点数: {len(df)}")
	print("数据预览:")
	print(df.head())

	return df


# ==================== 程序入口 ====================
if __name__ == "__main__":
	# 如果您没有实际数据，可以先运行以下代码生成示例数据
	# generate_sample_data("thrust_data.csv")

	# 运行主程序
	main()
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def parse_uml_file(file_name):
	"""解析UML文件"""
	uml_file = open(f'{file_name}.txt', 'r', encoding='utf-8')
	uml_read = uml_file.readlines()
	uml_file.close()

	variable_start_line = 0
	public_start_line = 0
	private_start_line = 0

	for i in range(len(uml_read)):
		if '类名' in uml_read[i]:
			uml_name = uml_read[i + 1].strip()
		if '变量' in uml_read[i]:
			variable_start_line = i + 1
		if '公有函数(+)' in uml_read[i]:
			public_start_line = i + 1
		if '私有函数(-)' in uml_read[i]:
			private_start_line = i + 1

	# 提取变量
	variable_list = []
	for i in range(variable_start_line, public_start_line - 1):
		line = uml_read[i].strip()
		if len(line) > 1:
			if '=' in line and ';' in line:
				# 处理有等号的情况
				var_parts = line.split('=')
				var_name_part = var_parts[0].strip()

				# 分割类型和变量名
				name_parts = var_name_part.split()
				if len(name_parts) >= 2:
					variable_type = name_parts[0]
					variable_name = name_parts[1]

					# 提取变量值
					var_value_part = '='.join(var_parts[1:])
					if ';' in var_value_part:
						variable_value = var_value_part.split(';')[0].strip()
					else:
						variable_value = var_value_part.strip()
				else:
					# 如果没有明确的类型和变量名，使用原始文本
					variable_type = 'unknown'
					variable_name = var_name_part
					variable_value = '='.join(var_parts[1:]).strip()
			elif '=' in line and ';' not in line:
				# 处理有等号的情况
				var_parts = line.split('=')
				var_name_part = var_parts[0].strip()

				# 分割类型和变量名
				name_parts = var_name_part.split()
				if len(name_parts) >= 2:
					variable_type = name_parts[0]
					variable_name = name_parts[1]
			elif '=' not in line and ';' in line and '}' not in line:
				# 处理没有等号的情况
				name_parts = line.split()
				if len(name_parts) >= 2:
					variable_type = name_parts[0]
					variable_name = name_parts[1].strip(';')
					variable_value = '/'
				else:
					variable_type = 'unknown'
					variable_name = line
					variable_value = '/'

			variable_list.append([variable_type, variable_name, variable_value])

	# 提取公有函数
	public_list = []
	for i in range(public_start_line, private_start_line - 1):
		line = uml_read[i].strip()
		if len(line) > 1 and '(' in line:
			# 分割函数名和参数
			func_parts = line.split('(')
			func_name_part = func_parts[0].strip()

			# 分割返回类型和函数名
			name_parts = func_name_part.split()
			if len(name_parts) >= 2:
				public_type = name_parts[0]
				public_name = name_parts[1]
			else:
				public_type = 'void'
				public_name = name_parts[0] if name_parts else line

			public_list.append([public_name, public_type])

	# 提取私有函数
	private_list = []
	for i in range(private_start_line, len(uml_read)):
		line = uml_read[i].strip()
		if len(line) > 1 and '(' in line:
			# 分割函数名和参数
			func_parts = line.split('(')
			func_name_part = func_parts[0].strip()

			# 分割返回类型和函数名
			name_parts = func_name_part.split()
			if len(name_parts) >= 2:
				private_type = name_parts[0]
				private_name = name_parts[1]
			else:
				private_type = 'void'
				private_name = name_parts[0] if name_parts else line

			private_list.append([private_name, private_type])

	return uml_name, variable_list, public_list, private_list


def draw_uml_diagram(uml_name, variable_list, public_list, private_list, output_file='uml_diagram.png'):
	"""绘制UML类图"""
	# 设置字体大小+格式
	font_size = 10
	title_font_size = 14
	paragraph_spacing = 15  # 段前间距
	line_spacing = 15  # 行间距

	# 计算每个部分的高度
	padding = 10  # 边界宽度
	class_box_height = 30
	var_box_height = len(variable_list) * 15 + line_spacing
	pub_box_height = len(public_list) * 15 + line_spacing
	pri_box_height = len(private_list) * 15 + line_spacing

	# 计算总高度
	total_height = class_box_height + var_box_height + pub_box_height + pri_box_height + padding * 2

	# 计算宽度（根据最长文本确定）
	max_width = 0

	# 检查类名宽度
	class_name_width = len(uml_name) * 8
	max_width = max(max_width, class_name_width)

	# 检查变量宽度
	for var in variable_list:
		var_text = f"{var[1]}: {var[0]} = {var[2]}"
		var_width = len(var_text) * 8
		max_width = max(max_width, var_width)

	# 检查公有函数宽度
	for func in public_list:
		func_text = f"+ {func[0]}(): {func[1]}"
		func_width = len(func_text) * 8
		max_width = max(max_width, func_width)

	# 检查私有函数宽度
	for func in private_list:
		func_text = f"- {func[0]}(): {func[1]}"
		func_width = len(func_text) * 8
		max_width = max(max_width, func_width)

	# 设置最小和最大宽度
	min_width = 400
	max_width = max(min_width, min(max_width, 800))

	# 创建图形
	fig, ax = plt.subplots(figsize=(max_width / 80, total_height / 80))
	ax.set_xlim(0, max_width)
	ax.set_ylim(0, total_height)
	ax.axis('off')

	# 绘制外部大框
	outer_rect = patches.Rectangle(
		(padding, padding), max_width - padding * 2, total_height - padding * 2,
		linewidth=2, edgecolor='black', facecolor='none'
	)
	ax.add_patch(outer_rect)

	# [绘制三个内部方框的分隔线]类名和变量之间的线
	class_var_y = total_height - padding - class_box_height
	ax.plot([padding, max_width - padding], [class_var_y, class_var_y], 'k-', linewidth=1)

	# [绘制三个内部方框的分隔线]变量和公有函数之间的线
	var_pub_y = class_var_y - var_box_height
	ax.plot([padding, max_width - padding], [var_pub_y, var_pub_y], 'k-', linewidth=1)

	# [绘制三个内部方框的分隔线]公有函数和私有函数之间的线
	pub_pri_y = var_pub_y - pub_box_height
	ax.plot([padding, max_width - padding], [pub_pri_y, pub_pri_y], 'k-', linewidth=1)

	# 添加类名（居中）
	ax.text(
		max_width / 2, class_var_y + class_box_height / 2,
		uml_name,
		ha='center', va='center',
		fontsize=title_font_size,
		fontweight='bold'
	)

	# 添加变量列表
	for i, var in enumerate(variable_list):
		y_pos = var_pub_y + var_box_height - line_spacing - i * paragraph_spacing
		var_text = f"{var[1]}: {var[0]} = {var[2]}"
		ax.text(
			15, y_pos,
			var_text,
			ha='left', va='center',
			fontsize=font_size - 1,
			fontfamily='monospace'
		)

	# 添加公有函数
	for i, func in enumerate(public_list):
		y_pos = pub_pri_y + pub_box_height - line_spacing - i * paragraph_spacing
		func_text = f"+ {func[0]}: {func[1]}"
		ax.text(
			20, y_pos,
			func_text,
			ha='left', va='center',
			fontsize=font_size - 1,
			fontfamily='monospace'
		)

	# 添加私有函数
	for i, func in enumerate(private_list):
		y_pos = padding + pri_box_height - line_spacing - i * paragraph_spacing
		func_text = f"- {func[0]}: {func[1]}"
		ax.text(
			20, y_pos,
			func_text,
			ha='left', va='center',
			fontsize=font_size - 1,
			fontfamily='monospace'
		)

	# 添加图表标题
	ax.text(
		max_width / 2, total_height - 5,
		f"{uml_text}的类图",
		ha='center', va='bottom',
		fontsize=16,
		fontweight='bold'
	)

	plt.tight_layout()
	plt.savefig(output_file, dpi=150, bbox_inches='tight')
	plt.show()

	print(f"UML图表已生成: {output_file}")


# 主程序
if __name__ == "__main__":
	# 设置中文字体和图形样式
	plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 支持中文显示
	plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

	# 解析文件
	uml_text = '无线电指令修正弹道'
	uml_name, variable_list, public_list, private_list = parse_uml_file(uml_text)

	# 打印信息（调试用）
	print('类名 =', uml_name)
	print(f'变量数量: {len(variable_list)}')
	print(f'公有函数数量: {len(public_list)}')
	print(f'私有函数数量: {len(private_list)}')

	# 绘制UML图
	draw_uml_diagram(uml_name, variable_list, public_list, private_list, f'{uml_text}.png')

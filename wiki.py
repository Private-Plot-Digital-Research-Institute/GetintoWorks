import os
import re
# 设置文件夹路径和目标文件名
folder_path = 'Java实用手册'  # 替换为实际的文件夹路径
target_file = 'B.md'  # 替换为实际的目标文件名
output_file = 'Java实用手册/wiki.md'  # 替换为实际的输出文件名

# 遍历文件夹中的文件
result_lines = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.md') and 'Java' in file:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if '# 'in line:
                        if "实例" in line:
                            continue
                        link_name = re.sub(r'^#+\s*', '', line)
                        result_line = \
                        f'##### '+f'{link_name.strip()}' + f'\n' \
                        + f'[[{file}#{line.strip()}|{link_name.strip()}{"笔记"}]]\n'  +f'\n'
                        # f'```'
                        # result_line = f'[[{file}|{line.lstrip("# ")}]]' +f'\n'


                        result_lines.append(result_line)

# 将结果写入输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(result_lines)
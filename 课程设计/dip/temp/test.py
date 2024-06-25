from pathlib import Path
import sys
import os

# 获取当前文件的绝对路径
file_path = Path(__file__).resolve()

# 获取当前文件的父目录
root_path = file_path.parent

# 如果根路径尚未添加到sys.path列表中，则将其添加
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 获取当前工作目录
current_working_directory = os.getcwd()

# 获取相对于当前工作目录的根目录的相对路径
ROOT = root_path.relative_to(Path.cwd())

print(file_path, root_path, current_working_directory, ROOT)
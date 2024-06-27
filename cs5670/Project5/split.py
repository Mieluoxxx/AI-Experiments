import os
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 获取data文件夹下所有文件夹名（即需要分类的类名）
file_path = './data'
flower_class = [cla for cla in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, cla))]

# 创建 训练集train 文件夹，并由类名在其目录下创建子目录
mkfile('./data/train')
for cla in flower_class:
    mkfile('./data/train/' + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile('./data/val')
for cla in flower_class:
    mkfile('./data/val/' + cla)

# 创建 测试集test 文件夹，并由类名在其目录下创建子目录
mkfile('./data/test')
for cla in flower_class:
    mkfile('./data/test/' + cla)

# 划分比例，训练集 : 验证集 : 测试集 = 7 : 2 : 1
train_rate = 0.7
val_rate = 0.2
test_rate = 0.1

# 遍历所有类别的全部图像并按比例分成训练集、验证集和测试集
for cla in flower_class:
    cla_path = os.path.join(file_path, cla)  # 某一类别的子目录
    images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    train_num = int(num * train_rate)
    val_num = int(num * val_rate)
    test_num = num - train_num - val_num

    train_images = random.sample(images, k=train_num)
    remaining_images = [img for img in images if img not in train_images]
    val_images = random.sample(remaining_images, k=val_num)
    test_images = [img for img in remaining_images if img not in val_images]

    for index, image in enumerate(images):
        image_path = os.path.join(cla_path, image)
        
        if image in train_images:
            new_path = os.path.join('./data/train', cla)
        elif image in val_images:
            new_path = os.path.join('./data/val', cla)
        else:
            new_path = os.path.join('./data/test', cla)

        copy(image_path, new_path)  # 将选中的图像复制到新路径
        print(f"\r[{cla}] processing [{index + 1}/{num}]", end="")  # processing bar
    print()

print("processing done!")

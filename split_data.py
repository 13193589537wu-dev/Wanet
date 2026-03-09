import os
import shutil
import random
from PIL import Image
from tqdm import tqdm

random.seed(42)

source_dir = r''  # ← 修改为你原始数据目录，含.jpg和.png
output_dir = r''  # ← 修改为输出格式化数据目录

# 输出子目录结构
splits = ['training', 'validation', 'testing']
for split in splits:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations', split), exist_ok=True)

# 匹配所有图像对
image_mask_pairs = []
for file in os.listdir(source_dir):
    if file.lower().endswith('.jpg'):
        base_name = os.path.splitext(file)[0]  # e.g., 00001
        mask_name = base_name + '_mask.png'
        mask_path = os.path.join(source_dir, mask_name)
        img_path = os.path.join(source_dir, file)
        if os.path.exists(mask_path):
            image_mask_pairs.append((img_path, mask_path))
        else:
            print(f"掩膜缺失: {mask_name}")
# 排序确保可分布采样
image_mask_pairs.sort()
total = len(image_mask_pairs)

# 计算划分数量
num_val = total // 10
num_test = total // 10
num_train = total - num_val - num_test

# 选择验证集和测试集（前中后各取1/3）
val_split = image_mask_pairs[:num_val//3] + \
            image_mask_pairs[total//2 - num_val//3:total//2] + \
            image_mask_pairs[-num_val//3:]
val_set = set(val_split)

remaining = [p for p in image_mask_pairs if p not in val_set]

test_split = remaining[:num_test//3] + \
             remaining[len(remaining)//2 - num_test//3:len(remaining)//2] + \
             remaining[-num_test//3:]
test_set = set(test_split)

train_split = [p for p in image_mask_pairs if p not in val_set and p not in test_set]

def save_pair(img_path, mask_path, split, index):
    """保存图像和掩码到对应目录"""
    base_name = f'ADE_{split}_{index:05d}'
    out_img = os.path.join(output_dir, 'images', split, base_name + '.jpg')
    out_mask = os.path.join(output_dir, 'annotations', split, base_name + '.png')
    shutil.copy(img_path, out_img)
    shutil.copy(mask_path, out_mask)

def augment_and_save_sequential(img_path, mask_path, counter):
    """保存原图和两个增强图，并使用连续编号命名"""
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    paths = []

    # 原图
    base_name = f'ADE_train_{counter:05d}'
    img.save(os.path.join(output_dir, 'images', 'training', base_name + '.jpg'))
    mask.save(os.path.join(output_dir, 'annotations', 'training', base_name + '.png'))
    paths.append(base_name)
    counter += 1

    # 水平翻转
    img_h = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask_h = mask.transpose(Image.FLIP_LEFT_RIGHT)
    base_name = f'ADE_train_{counter:05d}'
    img_h.save(os.path.join(output_dir, 'images', 'training', base_name + '.jpg'))
    mask_h.save(os.path.join(output_dir, 'annotations', 'training', base_name + '.png'))
    paths.append(base_name)
    counter += 1

    # 垂直翻转
    img_v = img.transpose(Image.FLIP_TOP_BOTTOM)
    mask_v = mask.transpose(Image.FLIP_TOP_BOTTOM)
    base_name = f'ADE_train_{counter:05d}'
    img_v.save(os.path.join(output_dir, 'images', 'training', base_name + '.jpg'))
    mask_v.save(os.path.join(output_dir, 'annotations', 'training', base_name + '.png'))
    paths.append(base_name)
    counter += 1

    return counter, paths

# 保存训练集（含增强）
train_index = 1
print("Processing training set with augmentation...")
for img_path, mask_path in tqdm(train_split):
    train_index, saved_names = augment_and_save_sequential(img_path, mask_path, train_index)
# 保存验证集
print("Processing validation set...")
for idx, (img_path, mask_path) in enumerate(tqdm(val_split)):
    save_pair(img_path, mask_path, 'validation', idx + 1)

# 保存测试集
print("Processing testing set...")
for idx, (img_path, mask_path) in enumerate(tqdm(test_split)):
    save_pair(img_path, mask_path, 'testing', idx + 1)

print("完成！数据已划分并保存到：", output_dir)

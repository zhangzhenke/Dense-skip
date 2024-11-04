import os
import numpy as np
from PIL import Image


def process_images(image_folder_path, label_folder_path):

    # 遍历image_folder_path下的所有文件
    for filename in os.listdir(image_folder_path):

        
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

            image_path = os.path.join(image_folder_path, filename)
            label_path = os.path.join(label_folder_path, filename)


            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file {label_path} does not exist. Skipping {filename}.")
                continue


            # 打开图片
            try:
                with Image.open(label_path) as img:
                    # 将图片转换为numpy数组
                    img_array = np.array(img)

                    # 检查第一个通道是否全为0
                    if img_array.sum() == 0:

                        print(f"Deleting image {image_path} and label {label_path}.")
                        os.remove(image_path)  # 删除图片文件
                        os.remove(label_path)  # 删除对应的标签文件

            except IOError:
                print(f"Cannot open {image_path}. Skipping.")



# 替换为你的图片文件夹路径和标签文件夹路径
image_folder_path = 'E:\\zhangzhenke\\Img_database\\dapi\\B01615B3\\img_224\\'
label_folder_path = 'E:\\zhangzhenke\\Img_database\\dapi\\B01615B3\\mask_224\\'


process_images(image_folder_path, label_folder_path)
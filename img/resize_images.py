import os

from torchvision import transforms as T
from PIL import Image
from torchvision.transforms import functional as F

def resize_images_in_folder(folder_path, output_folder_path, size=(512, 512)):

    # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)
            
            # 打开图片文件
            with Image.open(input_image_path) as img:

                # 使用最近邻插值方法调整图片大小
                resized_img = F.resize(img, size, interpolation=T.InterpolationMode.BICUBIC)
                
                # 保存调整大小后的图片
                resized_img.save(output_image_path)
                print(f"Resized and saved: {output_image_path}")



# 替换为你的图片文件夹路径和输出文件夹路径
input_folder_path = 'E:\\zhangzhenke\\Img_database\\dapi\\B01615B3\\img\\'
output_folder_path = 'E:\\zhangzhenke\\Img_database\\dapi\\B01615B3\\img_resize\\'


resize_images_in_folder(input_folder_path, output_folder_path)
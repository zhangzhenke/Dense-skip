import os
import PIL.Image
import numpy as np

from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None


def process_images(image_folder_path, output_folder_path, square_size=224):
   
    # 如果输出文件夹不存在，请创建该文件夹
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    
    # 列出文件夹中的所有图像
    images = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png', '.tiff', '.bmp', '.tif', '.jpeg'))]
    

    # 表示裁剪图像之间的重叠比例
    overlap = 0.2

    
    # 处理每个图像
    for k, image_name in enumerate(images):


        # 构造输入和输出文件的完整路径
        input_path = os.path.join(image_folder_path, image_name)

        
        # 加载图像
        image = Image.open(input_path)


        # 获取原始图像大小
        original_width, original_height = image.size

        
        # 计算每个方向所需的分割数量
        num_crops_width = int(original_width // (square_size - square_size * overlap)) + 1
        num_crops_height = int(original_height // (square_size - square_size * overlap)) + 1


        
        # 将图像裁剪成正方形并将其转换为灰度
        # 先w，后h
        for i in range(num_crops_height):
            for j in range(num_crops_width):

                # 计算分割的起始点
                start_x = j * int(square_size - square_size * overlap)
                start_y = i * int(square_size - square_size * overlap)

                
                # 从原始图像裁剪正方形
                # 右下角
                if (start_x + square_size) > (original_width) and (start_y + square_size) > (original_height):

                    cropped_image = image.crop((original_width - square_size, original_height - square_size, original_width, original_height))
                

                # 宽不够
                elif (start_x + square_size) > original_width:

                    cropped_image = image.crop((original_width - square_size, start_y, original_width, start_y + square_size))
                

                # 高不够
                elif (start_y + square_size) > original_height:

                    cropped_image = image.crop((start_x, original_height - square_size, start_x + square_size, original_height))


                # 中间区域
                else:
                    cropped_image = image.crop((start_x, start_y, start_x + square_size, start_y + square_size))
                

                # 保存裁剪和转换后的图像
                # 构造输出文件路径 
                output_file_path = os.path.join(output_folder_path, f"B01615B3_{k}_{i}_{j}.png")


        
                """# 将PIL图像转换为numpy数组
                cropped_image = np.array(cropped_image)

                # 归一化像素值到0-255范围
                # 由于Pillow读取的16-bit图像数据类型是uint16，我们需要先将其转换为float32，然后除以65535
                cropped_image  = (cropped_image / 65535.0).astype(np.float32)

                # 将归一化的单通道图像数据转换为三通道
                cropped_image = np.stack((cropped_image,) * 3, axis=-1)

                # 将归一化的浮点数数组转换回uint8
                cropped_image = (cropped_image * 255).astype(np.uint8)

                # 将numpy数组转换回PIL图像
                cropped_image = Image.fromarray(cropped_image)"""



                # 保存图像
                cropped_image.save(output_file_path)

                print(f"B01615B3_{k}_{i}_{j}.png 完成")



if __name__ == '__main__':

    image_folder_path = "E:\\zhangzhenke\\Img_database\\dapi\\B01615B3\\mask_resize\\"
    output_folder_path = "E:\\zhangzhenke\\Img_database\\dapi\\B01615B3\\mask_224\\"
    process_images(image_folder_path, output_folder_path)

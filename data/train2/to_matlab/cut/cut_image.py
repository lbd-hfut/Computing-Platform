import cv2
import os

# 获取当前工作目录
folder_path = os.getcwd()  # 当前文件夹路径

# 获取该文件夹下所有的图像文件
files = os.listdir(folder_path)

# 遍历所有文件
for file_name in files:
    # 获取文件的完整路径
    file_path = os.path.join(folder_path, file_name)
    
    # 获取文件扩展名
    _, ext = os.path.splitext(file_name)

    # 检查文件是否为指定格式的图像
    if ext.lower() in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        # 定义ROI的两个角点
        top_left = (18, 235)  # 左上角坐标
        bottom_right = (585, 408)  # 右下角坐标

        # 读取图像
        image = cv2.imread(file_path)

        # 使用提供的坐标裁剪图像
        roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 调整宽度和高度为16的倍数
        height, width = roi.shape[:2]
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        # 将ROI调整为16的倍数
        roi_resized = cv2.resize(roi, (new_width, new_height))

        # 覆盖原图像
        cv2.imwrite(file_path, roi_resized)

print("所有图像的ROI裁剪完成！")

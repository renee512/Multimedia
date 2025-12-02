import cv2
from extract import extract_plate_yolo
from split_char import split_char
from utils import cv_show, traverse_images
from retina import extract_retina
from color import detect_plate_color
import os

def main(image_path):

    
    # 读取原始图像
    origin_image = cv2.imread(image_path)
    
    # 第一步：使用YOLOv8检测车牌
    plate_image = extract_plate_yolo(origin_image)
    
    # 准备临时文件路径
    filename = os.path.basename(image_path)
    tmp_dir = ".tmp"
    path = os.path.join(tmp_dir, filename)
    
    # 创建临时目录
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    plate_detected = False
    
    if plate_image is not None:
        # YOLOv8检测成功
        cv2.imwrite(path, plate_image)
        plate_detected = True
        print("YOLOv8检测到车牌")
    else:
        # YOLOv8检测失败，尝试RetinaNet
        print("YOLOv8未检测到车牌，尝试RetinaNet...")
        plate_retina = extract_retina(image_path, path)
        if plate_retina:
            plate_detected = True
            print("RetinaNet检测到车牌")
        else:
            # 两种检测方法都失败，尝试直接OCR
            print("检测模型均失败，尝试直接OCR识别...")
            chars = split_char(image_path)
            if not chars:
                print("没有检测到车牌！！！")
                return None
            else:
                # 直接OCR成功
                plate_color = detect_plate_color(origin_image)
                return format_plate_result(chars, plate_color)
    
    if plate_detected:
        chars = split_char(path)
        
        if not chars:
            # 如果裁剪后的车牌识别失败，尝试识别原图
            print("裁剪车牌识别失败，尝试原图识别...")
            chars = split_char(image_path)
    
    # 第三步：检测车牌颜色和格式化结果
    if chars:
        plate_color = detect_plate_color(cv2.imread(path if plate_detected else image_path))
        return format_plate_result(chars, plate_color)
    else:
        print("字符识别失败")
        return None

def format_plate_result(chars, plate_color):
    """
    格式化车牌结果
    """
    # 添加分隔点
    if len(chars) >= 3 and chars[2] != '·':
        formatted_chars = chars[0:2] + '·' + chars[2:]
    else:
        formatted_chars = chars
    
    # 新能源车牌特殊处理
    if len(formatted_chars) == 9:
        plate_color = "Green"
    
    result = {
        'plate_number': formatted_chars,
        'plate_color': plate_color,
        'success': True
    }
    
    print(f"识别结果: 车牌{plate_color}色, 号码: {formatted_chars}")
    return result

if __name__ == "__main__":
    # 测试单张图片
    result = main("test_images/001.jpg")
    if result:
        print(f"最终结果: {result}")
    else:
        print("识别失败")
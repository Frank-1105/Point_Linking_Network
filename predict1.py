# from cv2 import cv2
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# import cv2
from torchvision.transforms import ToTensor
from matplotlib.patches import Rectangle
import glob
from pathlib import Path

from PLNnet import pretrained_inception, inceptionresnetv2
from validate import extract_boxes_from_targets, decode_predictions, non_max_suppression

# from new_resnet import pretrained_inception

# from draw_rectangle import draw

# voc数据集的类别信息，这里转换成字典形式
classes = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8,
           "cow": 9, "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15,
           "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

# 测试图片的路径
# img_root = r"E:\datasets\pascalvoc\pascalvoc\VOCdevkit\VOC2007\JPEGImages\002621.jpg"
img_root = r"C:\Users\ZHENGZHIQIAN\Desktop\20250507210205.jpg"
# # 网络模型
# model = resnet50()

# VOC数据集的类别信息
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_NUM = len(VOC_CLASSES)

# 颜色映射，为每个类别分配一个颜色
COLOR_MAP = {
    i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(CLASS_NUM)
}

class PredictConfig:
    """预测配置类，集中管理预测参数"""
    def __init__(self):
        # 检测阈值设置
        self.p_threshold = 0.2        # 点存在性阈值
        self.score_threshold = 0.15    # 得分阈值
        self.nms_threshold = 0.2      # NMS置信度阈值
        self.iou_threshold = 0.3      # IOU阈值
        
        # 图像处理设置
        self.image_size = (448, 448)  # 输入图像大小
        self.mean = (123, 117, 104)   # 图像均值 (RGB)
        
        # 设备设置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 绘图设置
        self.show_scores = True       # 显示得分
        self.show_class = True        # 显示类别
        self.line_thickness = 2       # 边框粗细
        self.font_scale = 0.75        # 字体大小
        self.gt_box_color = (0, 255, 0)  # 真实框颜色 (绿色)
        self.pred_box_color = (0, 0, 255)  # 预测框颜色 (红色)

class Predictor:
    """目标检测预测器"""
    
    def __init__(self, model, config=None):
        """
        初始化预测器
        
        参数:
            model: 训练好的模型
            config: 预测配置
        """
        self.model = model
        self.model.eval()  # 设置为评估模式
        self.config = config if config else PredictConfig()
        
        # 确保模型在正确的设备上
        self.device = self.config.device
        self.model = self.model.to(self.device)
        
    def compute_area(self, branch, j, i, grid_size=14):
        """
        计算搜索区域
        
        参数:
            branch: 分支编号 (0=左下，1=左上，2=右下，3=右上)
            j: 列索引
            i: 行索引
            grid_size: 网格大小
            
        返回:
            area: 搜索区域 [[x_min,x_max], [y_min,y_max]]
        """
        area = [[], []]
        if branch == 0:
            # 左下角
            area = [[0, j+1], [i, grid_size]]
        elif branch == 1:
            # 左上角
            area = [[0, j+1], [0, i+1]]
        elif branch == 2:
            # 右下角
            area = [[j, grid_size], [i, grid_size]]
        elif branch == 3:
            # 右上角
            area = [[j, grid_size], [0, i+1]]
        return area
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        参数:
            image_path: 图像路径
            
        返回:
            img: 预处理后的图像张量
            original_image: 原始图像
            h, w: 原始图像的高宽
        """
        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        # 保存原始尺寸
        h, w, _ = original_image.shape
        
        # 调整图像大小
        image = cv2.resize(original_image, self.config.image_size)
        
        # BGR 转 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 减去均值
        image_normalized = image_rgb - np.array(self.config.mean, dtype=np.float32)
        
        # 转换为张量
        transform = ToTensor()
        img_tensor = transform(image_normalized)
        
        # 添加批次维度
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, original_image, h, w
    
    def predict_single_image(self, image_path, gt_boxes=None):
        """
        对单个图像进行预测
        
        参数:
            image_path: 图像路径
            gt_boxes: 真实标注框 [x_min, y_min, x_max, y_max, conf, class_id]
            
        返回:
            bboxes: 预测的边界框
            image: 绘制了边界框的图像
        """
        # 预处理图像
        img_tensor, original_image, orig_h, orig_w = self.preprocess_image(image_path)
        
        # 前向传播
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # 创建一个副本用于绘制
        display_image = original_image.copy()
        
        # 针对四个分支进行解码
        all_boxes = []
        for i, pred in enumerate(predictions):
            # 调整张量维度
            pred_permuted = pred.permute(0, 2, 3, 1)[0]  # 只处理batch中的第一张图片
            
            # 使用优化后的解码函数
            boxes = decode_predictions(
                pred_permuted, 
                branch=i,
                p_threshold=self.config.p_threshold, 
                score_threshold=self.config.score_threshold,
                device=self.device
            )
            
            if boxes.size(0) > 0:
                all_boxes.append(boxes)
        
        # 如果有检测结果，合并并应用NMS
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            bboxes = non_max_suppression(
                all_boxes, 
                nms_threshold=self.config.nms_threshold, 
                iou_threshold=self.config.iou_threshold
            )
        else:
            bboxes = []
            
        # 将预测框映射回原始图像尺寸
        scale_w = orig_w / self.config.image_size[0]
        scale_h = orig_h / self.config.image_size[1]
        
        # 绘制真实标注框 (如果有)
        if gt_boxes:
            self.draw_boxes(display_image, gt_boxes, is_gt=True)
        
        # 绘制预测框
        self.draw_boxes(display_image, bboxes, is_gt=False, 
                       scale_w=scale_w, scale_h=scale_h)
        
        return bboxes, display_image
    
    def draw_boxes(self, image, boxes, is_gt=False, scale_w=1.0, scale_h=1.0):
        """
        在图像上绘制边界框
        
        参数:
            image: 要绘制的图像
            boxes: 边界框列表
            is_gt: 是否为真实标注框
            scale_w, scale_h: 宽高比例尺
        """
        if not boxes:
            return
            
        box_color = self.config.gt_box_color if is_gt else self.config.pred_box_color
        thickness = self.config.line_thickness
        font_scale = self.config.font_scale
        
        for box in boxes:
            # 提取坐标和标签
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
                
            x1, y1, x2, y2 = int(box[0] * scale_w), int(box[1] * scale_h), \
                             int(box[2] * scale_w), int(box[3] * scale_h)
            score = box[4] if len(box) > 4 else 1.0
            class_id = int(box[5]) if len(box) > 5 else -1
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
            
            # 绘制标签
            if (class_id >= 0 and self.config.show_class) or self.config.show_scores:
                label_parts = []
                if class_id >= 0 and self.config.show_class:
                    label_parts.append(VOC_CLASSES[class_id])
                if self.config.show_scores:
                    label_parts.append(f"{score:.2f}")
                    
                label = " ".join(label_parts)
                cv2.putText(image, label, (x1, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness)
    
    def predict_directory(self, image_dir, gt_file=None, output_dir=None, limit=None):
        """
        预测目录中的所有图像
        
        参数:
            image_dir: 图像目录路径
            gt_file: 真实标注文件路径
            output_dir: 输出目录路径，如果为None则不保存结果
            limit: 处理图像的最大数量，如果为None则处理所有图像
            
        返回:
            results: 预测结果字典 {图像路径: (边界框, 处理后图像)}
        """
        # 获取图像文件列表
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) + \
                     sorted(glob.glob(os.path.join(image_dir, "*.jpeg"))) + \
                     sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        if limit:
            image_files = image_files[:limit]
            
        # 读取真实标注 (如果有)
        gt_data = {}
        if gt_file and os.path.exists(gt_file):
            # 获取所有的真实标注
            all_gt_boxes = extract_boxes_from_targets(gt_file, 0, float('inf'))
            
            # 按图像组织真实标注
            current_image = None
            current_boxes = []
            
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        img_name = parts[0]
                        
                        # 如果是新图像
                        if img_name != current_image:
                            # 保存上一个图像的标注
                            if current_image:
                                gt_data[current_image] = current_boxes
                            
                            current_image = img_name
                            current_boxes = []
                            
                        # 解析该行的所有边界框
                        box_parts = parts[1:]
                        num_boxes = len(box_parts) // 5
                        
                        for i in range(num_boxes):
                            xmin = float(box_parts[i * 5])
                            ymin = float(box_parts[i * 5 + 1])
                            xmax = float(box_parts[i * 5 + 2])
                            ymax = float(box_parts[i * 5 + 3])
                            class_id = int(box_parts[i * 5 + 4])
                            
                            current_boxes.append([xmin, ymin, xmax, ymax, 1.0, class_id])
                
                # 保存最后一个图像的标注
                if current_image:
                    gt_data[current_image] = current_boxes
        
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个图像
        results = {}
        for i, image_file in enumerate(image_files):
            print(f"处理图像 {i+1}/{len(image_files)}: {image_file}")
            
            # 获取图像文件名
            img_basename = os.path.basename(image_file)
            
            # 获取对应的真实标注
            gt_boxes = None
            if img_basename in gt_data:
                gt_boxes = gt_data[img_basename]
            
            # 进行预测
            bboxes, result_image = self.predict_single_image(image_file, gt_boxes)
            
            # 保存结果
            results[image_file] = (bboxes, result_image)
            
            if output_dir:
                output_path = os.path.join(output_dir, img_basename)
                cv2.imwrite(output_path, result_image)
                print(f"结果已保存至: {output_path}")
        
        return results
    
    def display_result(self, image, figsize=(10, 8)):
        """
        显示结果图像
        
        参数:
            image: 要显示的图像
            figsize: 图像尺寸
        """
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


def init_model(model_path, model_type="inceptionresnetv2"):
    """
    初始化模型
    
    参数:
        model_path: 模型权重路径
        model_type: 模型类型
        
    返回:
        model: 初始化后的模型
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == "inceptionresnetv2":
        from PLNnet import inceptionresnetv2
        model = inceptionresnetv2(num_classes=CLASS_NUM, pretrained='imagenet').to(device)
    elif model_type == "inception":
        from PLNnet import pretrained_inception
        model = pretrained_inception().to(device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"加载模型成功，训练轮次: {epoch}")
    
    return model


def main():
    # 配置路径
    model_path = r"E:\study\2025spring\baoyan\Paper reproduction\PLN\PLN\results\5\pln_latest.pth"
    image_dir = r"./test_images"  # 测试图像目录
    output_dir = r"./test_results"  # 输出结果目录
    gt_file = r"C:\Users\ZHENGZHIQIAN\Desktop\PLN\voctest.txt"  # 真实标注文件路径，如果有的话
    
    # 初始化模型
    model = init_model(model_path)
    
    # 创建配置
    config = PredictConfig()
    
    # 创建预测器
    predictor = Predictor(model, config)
    
    # 处理单张图像示例
    single_image_path = r"C:\Users\ZHENGZHIQIAN\Desktop\20250507210205.jpg"
    if os.path.exists(single_image_path):
        print(f"处理单张图像: {single_image_path}")
        bboxes, result_image = predictor.predict_single_image(single_image_path)
        
        # 保存结果
        output_path = os.path.join(".", "single_result.jpg")
        cv2.imwrite(output_path, result_image)
        print(f"单张图像结果已保存至: {output_path}")
        
        # 显示检测结果
        for i, box in enumerate(bboxes):
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            print(f"检测框 {i+1}: "
                  f"坐标=({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}), "
                  f"类别={VOC_CLASSES[int(box[5])]}, 置信度={box[4]:.3f}")
    
    # 处理整个目录
    if os.path.exists(image_dir):
        print(f"\n处理目录: {image_dir}")
        results = predictor.predict_directory(image_dir, gt_file, output_dir)
        print(f"共处理 {len(results)} 张图像")


if __name__ == "__main__":
    main()

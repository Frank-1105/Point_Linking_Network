import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import time
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 导入 torchmetrics 用于评估
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# 导入项目模块
from PLNdata import PLNDataset
from PLNLoss import PLNLoss
from PLNnet import pretrained_inception
from validate import (calculate_iou, compute_area, decode_predictions, non_max_suppression, 
                      extract_boxes_from_predictions, extract_boxes_from_targets)

# VOC类别信息
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_NUM = len(VOC_CLASSES)


def convert_to_torchmetrics_format(detections, ground_truths, batch_size=1):
    """
    将检测结果和真实标注转换为torchmetrics支持的格式
    
    参数:
        detections: 检测结果列表 [[x1,y1,x2,y2,score,class_id], ...]
        ground_truths: 真实标注列表 [[x1,y1,x2,y2,score,class_id], ...]
        batch_size: 批次大小
        
    返回:
        preds: torchmetrics格式的预测结果
        target: torchmetrics格式的真实标注
    """
    preds = []
    targets = []
    
    # 按批次整理数据
    for b in range(batch_size):
        # 构建预测结果字典
        pred_dict = {
            'boxes': [],
            'scores': [],
            'labels': []
        }
        
        # 构建目标字典
        target_dict = {
            'boxes': [],
            'labels': []
        }
        
        # 加入检测结果
        for det in detections:
            if isinstance(det, torch.Tensor):
                det = det.cpu().numpy().tolist()
            pred_dict['boxes'].append(det[:4])
            pred_dict['scores'].append(det[4])
            pred_dict['labels'].append(int(det[5]) + 1)  # torchmetrics要求类别从1开始
        
        # 加入真实标注
        for gt in ground_truths:
            if isinstance(gt, torch.Tensor):
                gt = gt.cpu().numpy().tolist()
            target_dict['boxes'].append(gt[:4])
            target_dict['labels'].append(int(gt[5]) + 1)  # torchmetrics要求类别从1开始
        
        # 转换为tensor
        if pred_dict['boxes']:
            pred_dict['boxes'] = torch.tensor(pred_dict['boxes'])
            pred_dict['scores'] = torch.tensor(pred_dict['scores'])
            pred_dict['labels'] = torch.tensor(pred_dict['labels'])
            preds.append(pred_dict)
        else:
            # 如果没有检测到任何物体，创建空张量
            pred_dict['boxes'] = torch.zeros((0, 4))
            pred_dict['scores'] = torch.zeros(0)
            pred_dict['labels'] = torch.zeros(0, dtype=torch.int)
            preds.append(pred_dict)
            
        if target_dict['boxes']:
            target_dict['boxes'] = torch.tensor(target_dict['boxes'])
            target_dict['labels'] = torch.tensor(target_dict['labels'])
            targets.append(target_dict)
        else:
            # 如果没有真实标注，创建空张量
            target_dict['boxes'] = torch.zeros((0, 4))
            target_dict['labels'] = torch.zeros(0, dtype=torch.int)
            targets.append(target_dict)
    
    return preds, targets


def evaluate_model(model, val_loader, device, config=None):
    """
    使用torchmetrics评估模型性能
    
    参数:
        model: 要评估的模型
        val_loader: 验证数据加载器
        device: 设备（'cuda'或'cpu'）
        config: 配置参数
        
    返回:
        metrics: 评估指标字典
    """
    if config is None:
        config = {
            'p_threshold': 0.3,
            'score_threshold': 0.2,
            'nms_threshold': 0.3,
            'iou_threshold': 0.5,
            'iou_thresholds': [0.5, 0.75]  # 计算mAP时使用的IoU阈值
        }
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化torchmetrics指标计算器
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_thresholds=config['iou_thresholds'],
        rec_thresholds=torch.linspace(0, 1, 101),  # 101点插值
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True  # 计算每个类别的指标
    )
    
    # 记录推理时间
    total_time = 0
    total_images = 0
    
    print("开始评估模型性能...")
    with torch.no_grad():
        start_index = 0
        for images, targets in tqdm(val_loader, desc="推理进度"):
            batch_size = images.size(0)
            total_images += batch_size
            end_index = start_index + batch_size
            
            # 将数据移到指定设备
            images = images.to(device)
            
            # 从文件中读取当前批次对应的目标框
            target_boxes = extract_boxes_from_targets("voctestceshi1.txt", start_index, end_index)
            
            # 记录推理开始时间
            start_time = time.time()
            
            # 模型推理
            predictions = model(images)
            
            # 记录推理结束时间
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 存储当前批次的所有检测结果
            batch_detections = []
            
            # 获取预测框
            for b in range(batch_size):
                # 提取单个样本的预测
                batch_preds = [p[b:b+1] for p in predictions]
                
                # 从预测中提取框
                boxes = extract_boxes_from_predictions(batch_preds, device, config)
                
                # 添加到结果列表
                batch_detections.extend(boxes)
            
            # 格式转换为torchmetrics使用的格式
            preds, gt_targets = convert_to_torchmetrics_format(batch_detections, target_boxes, batch_size=1)
            
            # 更新指标
            metric.update(preds, gt_targets)
            
            start_index = end_index
    
    # 计算平均推理时间
    avg_time_per_image = total_time / max(total_images, 1)
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
    
    print(f"平均每张图像推理时间: {avg_time_per_image*1000:.2f}ms (FPS: {fps:.2f})")
    
    # 计算最终指标
    metrics = metric.compute()
    
    # 转换为可处理的字典格式
    result_dict = {
        'per_class': {},
        'overall': {}
    }
    
    # 处理总体指标
    result_dict['overall']['mAP@0.5'] = metrics['map_50'].item()
    result_dict['overall']['mAP@0.75'] = metrics['map_75'].item()
    result_dict['overall']['mAP'] = metrics['map'].item()  # 所有IoU阈值的平均
    result_dict['overall']['mAR@1'] = metrics['mar_1'].item()
    result_dict['overall']['mAR@10'] = metrics['mar_10'].item()
    result_dict['overall']['mAR@100'] = metrics['mar_100'].item()
    result_dict['overall']['inference_time_ms'] = avg_time_per_image * 1000
    result_dict['overall']['fps'] = fps
    
    # 处理类别指标（如果可用）
    if 'map_per_class' in metrics:
        for c in range(CLASS_NUM):
            class_name = VOC_CLASSES[c]
            result_dict['per_class'][class_name] = {}
            
            # 注意: torchmetrics中类别从1开始，而我们的类别从0开始
            class_idx = c + 1
            
            # 保存每个类别的AP (map_50)
            if class_idx < len(metrics['map_per_class']) and not torch.isnan(metrics['map_per_class'][class_idx]):
                result_dict['per_class'][class_name]['AP@0.5'] = metrics['map_50_per_class'][class_idx].item()
            else:
                result_dict['per_class'][class_name]['AP@0.5'] = 0.0
            
            # 保存每个类别的AP (map_75)
            if class_idx < len(metrics['map_per_class']) and not torch.isnan(metrics['map_75_per_class'][class_idx]):
                result_dict['per_class'][class_name]['AP@0.75'] = metrics['map_75_per_class'][class_idx].item()
            else:
                result_dict['per_class'][class_name]['AP@0.75'] = 0.0
            
            # 保存每个类别的AP (平均)
            if class_idx < len(metrics['map_per_class']) and not torch.isnan(metrics['map_per_class'][class_idx]):
                result_dict['per_class'][class_name]['AP'] = metrics['map_per_class'][class_idx].item()
            else:
                result_dict['per_class'][class_name]['AP'] = 0.0
    
    return result_dict


def plot_precision_recall_curve_from_torchmetrics(metric, save_path=None):
    """
    使用torchmetrics计算的结果绘制精确率-召回率曲线
    
    参数:
        metric: torchmetrics.detection.MeanAveragePrecision实例
        save_path: 保存路径，如果为None则显示图像
    """
    # 获取计算结果
    metrics = metric.compute()
    
    plt.figure(figsize=(10, 8))
    
    # 获取PR曲线数据
    if 'precision' in metrics and 'recall' in metrics:
        # 获取每个类别的precision和recall
        precision = metrics['precision']
        recall = metrics['recall']
        
        # 确保数据有效
        if precision.dim() >= 3 and recall.dim() >= 2:
            # 遍历每个类别 (注意: 类别索引从1开始)
            for c in range(1, CLASS_NUM + 1):
                if c < precision.shape[1]:
                    # 获取IoU=0.5时的PR曲线 (第一个IoU阈值)
                    class_precision = precision[0, c].cpu().numpy()
                    class_recall = recall[0, c].cpu().numpy()
                    
                    # 计算AP
                    valid_mask = ~np.isnan(class_precision) & ~np.isnan(class_recall)
                    if np.any(valid_mask):
                        ap = metrics['map_50_per_class'][c].item() if c < len(metrics['map_50_per_class']) else 0
                        
                        # 绘制PR曲线
                        plt.plot(class_recall[valid_mask], class_precision[valid_mask], 
                                 label=f'{VOC_CLASSES[c-1]} (AP={ap:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (IoU=0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"已保存精确率-召回率曲线至 {save_path}")
    else:
        plt.show()


def main(args=None):
    """
    主函数
    
    参数:
        args: 参数字典，如果为None则使用默认值
    """
    # 验证函数参数配置
    if args is None:
        args = {
            'model_path': r"E:\study\2025spring\baoyan\Paper reproduction\PLN\PLN\results\5\pln_latest.pth",  # 模型权重路径
            'test_file': 'voctestceshi.txt',              # 测试集文件
            'img_root': 'E:\datasets\pascalvoc\pascalvoc\VOCdevkit\VOC2007\JPEGImages/',  # 图像根目录
            'batch_size': 4,                         # 批次大小
            'num_workers': 4,                        # 数据加载线程数
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 设备
            'plot_pr_curve': True,                   # 是否绘制PR曲线
            'save_results': True,                    # 是否保存结果
            'config': {                              # 评估配置
                'p_threshold': 0.2,                  # 点存在性阈值
                'score_threshold': 0.2,              # 得分阈值
                'nms_threshold': 0.3,                # NMS阈值  小于舍去
                'iou_threshold': 0.3,                # IoU阈值  大于则在NMS中舍去
                'iou_thresholds': [0.5, 0.75]        # 用于计算mAP的IoU阈值
            }
        }
    
    # 打印验证设置
    print("\n验证设置:")
    print(f"模型路径: {args['model_path']}")
    print(f"测试集文件: {args['test_file']}")
    print(f"图像根目录: {args['img_root']}")
    print(f"批次大小: {args['batch_size']}")
    print(f"设备: {args['device']}")
    
    # 加载模型
    print("\n正在加载模型...")
    model = pretrained_inception().to(args['device'])
    
    # 加载模型权重
    checkpoint = torch.load(args['model_path'], map_location=args['device'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"成功加载模型，轮次: {checkpoint['epoch']}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载测试数据集
    print("\n正在加载测试数据集...")
    test_dataset = PLNDataset(
        img_root=args['img_root'],
        list_file=args['test_file'],
        train=False,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)} 张图像")
    
    # 评估模型性能
    start_time = time.time()
    metrics = evaluate_model(model, test_loader, args['device'], args['config'])
    eval_time = time.time() - start_time
    
    # 打印总评估时间
    print(f"\n评估完成，总耗时: {eval_time:.2f}秒")
    
    # 打印mAP和总体指标
    print("\n整体性能:")
    for metric, value in metrics['overall'].items():
        print(f"{metric}: {value:.4f}")
    
    # 打印各类别的AP
    print("\n各类别AP@0.5:")
    for c in range(CLASS_NUM):
        class_name = VOC_CLASSES[c]
        ap = metrics['per_class'][class_name]['AP@0.5']
        print(f"{class_name}: {ap:.4f}")
    
    # 保存结果
    if args['save_results']:
        # 创建结果目录
        os.makedirs('eval_results', exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        # 保存评估指标
        result_path = f"eval_results/metrics_torchmetrics_{timestamp}.json"
        
        # 将结果转换为JSON可序列化格式
        json_metrics = json.dumps(metrics, indent=4)
        
        with open(result_path, 'w') as f:
            f.write(json_metrics)
        
        print(f"\n评估指标已保存至 {result_path}")
        
        # 如果需要绘制PR曲线，需要重新收集数据
        if args['plot_pr_curve']:
            # 初始化torchmetrics指标计算器
            metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_thresholds=args['config']['iou_thresholds'],
                rec_thresholds=torch.linspace(0, 1, 101),  # 101点插值
                max_detection_thresholds=[1, 10, 100],
                class_metrics=True  # 计算每个类别的指标
            )
            
            # 收集数据
            with torch.no_grad():
                start_index = 0
                for images, targets in tqdm(test_loader, desc="收集PR曲线数据"):
                    batch_size = images.size(0)
                    end_index = start_index + batch_size
                    
                    # 将数据移到指定设备
                    images = images.to(args['device'])
                    
                    # 读取真实框
                    target_boxes = extract_boxes_from_targets("voctestceshi1.txt", start_index, end_index)
                    
                    # 模型推理
                    predictions = model(images)
                    
                    # 获取预测框
                    batch_detections = []
                    for b in range(batch_size):
                        batch_preds = [p[b:b+1] for p in predictions]
                        boxes = extract_boxes_from_predictions(batch_preds, args['device'], args['config'])
                        batch_detections.extend(boxes)
                    
                    # 转换格式并更新指标
                    preds, gt_targets = convert_to_torchmetrics_format(batch_detections, target_boxes, batch_size=1)
                    metric.update(preds, gt_targets)
                    
                    start_index = end_index
            
            # 绘制PR曲线
            pr_curve_path = f"eval_results/pr_curve_torchmetrics_{timestamp}.png"
            plot_precision_recall_curve_from_torchmetrics(metric, save_path=pr_curve_path)
    
    return metrics


if __name__ == "__main__":
    main() 
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
from collections import defaultdict
import matplotlib.pyplot as plt

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


def calculate_ap_per_class(detections, ground_truths, class_idx, iou_threshold=0.5):
    """
    计算特定类别的平均精度(AP)
    
    参数:
        detections: 检测结果列表 [[x1,y1,x2,y2,score,class_id], ...]
        ground_truths: 真实标注列表 [[x1,y1,x2,y2,score,class_id], ...]
        class_idx: 当前类别索引
        iou_threshold: IoU阈值
        
    返回:
        ap: 该类别的平均精度
        precision: 精度数组
        recall: 召回率数组
    """
    # 过滤当前类别的预测和真实标注
    class_dets = [d for d in detections if d[5] == class_idx]
    class_gt = [g for g in ground_truths if g[5] == class_idx]
    
    # 如果没有真实标注，AP为0
    if len(class_gt) == 0:
        return 0.0, np.array([]), np.array([])
    
    # 按置信度排序预测结果
    class_dets = sorted(class_dets, key=lambda x: x[4], reverse=True)
    
    # 初始化TP和FP数组
    tp = np.zeros(len(class_dets))
    fp = np.zeros(len(class_dets))
    
    # 标记已匹配的真实标注
    gt_matched = [False] * len(class_gt)
    
    # 遍历每个检测结果
    for i, det in enumerate(class_dets):
        # 找到与当前检测结果IoU最大的真实标注
        max_iou = -float('inf')
        max_idx = -1
        
        for j, gt in enumerate(class_gt):
            # 如果该真实标注已经被匹配，跳过
            if gt_matched[j]:
                continue
                
            # 计算IoU
            iou = calculate_iou(det[:4], gt[:4])
            
            # 更新最大IoU
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        
        # 判断是否为TP
        if max_idx != -1 and max_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1
    
    # 计算累积TP和FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # 计算精度和召回率
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall = cum_tp / len(class_gt)
    print("类别：",class_idx,"target数量：",len(class_gt),"detect数量：",len(class_dets))
    print("累计TP：",cum_tp,"累计FP：",cum_fp)
    # 计算AP（按照PASCAL VOC方法）
    # 对召回率进行插值
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11
    
    return ap, precision, recall


def calculate_map(all_detections, all_ground_truths, iou_thresholds=None):
    """
    计算多个IoU阈值下的mAP和每个类别的AP
    
    参数:
        all_detections: 所有检测结果的列表
        all_ground_truths: 所有真实标注的列表
        iou_thresholds: IoU阈值列表，如果为None则使用[0.5]
        
    返回:
        result_dict: 包含各种指标的字典
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # 按类别和IoU阈值初始化结果字典
    result_dict = {
        'per_class': {},
        'overall': {}
    }
    
    # 初始化各IoU阈值下的AP列表
    for threshold in iou_thresholds:
        result_dict['overall'][f'mAP@{threshold}'] = 0.0
        
    # 记录各类别不同阈值下的指标
    for c in range(CLASS_NUM):
        class_name = VOC_CLASSES[c]
        result_dict['per_class'][class_name] = {}
        
        # 各IoU阈值下的AP
        for threshold in iou_thresholds:
            ap, precision, recall = calculate_ap_per_class(
                all_detections, all_ground_truths, c, threshold
            )
            
            result_dict['per_class'][class_name][f'AP@{threshold}'] = ap
            
            # 计算F1分数
            if len(precision) > 0 and len(recall) > 0:
                # 找到最大F1对应的位置
                f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                max_f1_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
                
                if len(f1_scores) > 0:
                    result_dict['per_class'][class_name][f'F1@{threshold}'] = f1_scores[max_f1_idx]
                    result_dict['per_class'][class_name][f'Precision@{threshold}'] = precision[max_f1_idx]
                    result_dict['per_class'][class_name][f'Recall@{threshold}'] = recall[max_f1_idx]
                else:
                    result_dict['per_class'][class_name][f'F1@{threshold}'] = 0.0
                    result_dict['per_class'][class_name][f'Precision@{threshold}'] = 0.0
                    result_dict['per_class'][class_name][f'Recall@{threshold}'] = 0.0
            else:
                result_dict['per_class'][class_name][f'F1@{threshold}'] = 0.0
                result_dict['per_class'][class_name][f'Precision@{threshold}'] = 0.0
                result_dict['per_class'][class_name][f'Recall@{threshold}'] = 0.0
                
            # 累加到总AP
            result_dict['overall'][f'mAP@{threshold}'] += ap
    
    # 计算mAP（各类别AP的平均值）
    for threshold in iou_thresholds:
        result_dict['overall'][f'mAP@{threshold}'] /= CLASS_NUM
        
    # 计算总体的精确率、召回率和F1分数
    result_dict['overall']['Precision'] = np.mean([
        result_dict['per_class'][VOC_CLASSES[c]]['Precision@0.5'] 
        for c in range(CLASS_NUM)
    ])
    
    result_dict['overall']['Recall'] = np.mean([
        result_dict['per_class'][VOC_CLASSES[c]]['Recall@0.5'] 
        for c in range(CLASS_NUM)
    ])
    
    result_dict['overall']['F1'] = np.mean([
        result_dict['per_class'][VOC_CLASSES[c]]['F1@0.5'] 
        for c in range(CLASS_NUM)
    ])
    
    return result_dict


# def evaluate_model(model, val_loader, device, config=None):
#     """
#     评估模型性能，计算各种指标
#
#     参数:
#         model: 要评估的模型
#         val_loader: 验证数据加载器
#         device: 设备（'cuda'或'cpu'）
#         config: 配置参数
#
#     返回:
#         metrics: 评估指标字典
#     """
#     if config is None:
#         config = {
#             'p_threshold': 0.1,
#             'score_threshold': 0.1,
#             'nms_threshold': 0.1,
#             'iou_threshold': 0.4,
#             'iou_thresholds': [0.5, 0.75]  # 计算mAP时使用的IoU阈值
#         }
#
#     # 设置模型为评估模式
#     model.eval()
#
#     # 收集所有预测结果和真实标注
#     all_detections = []
#     all_ground_truths = []
#
#     # 记录推理时间
#     total_time = 0
#     total_images = 0
#
#     print("开始评估模型性能...")
#     with torch.no_grad():
#         for images, targets in tqdm(val_loader, desc="推理进度"):
#             batch_size = images.size(0)
#             total_images += batch_size
#
#             # 将数据移到指定设备
#             images = images.to(device)
#             targets = targets.to(device)
#
#             # 记录推理开始时间
#             start_time = time.time()
#
#             # 模型推理
#             predictions = model(images)
#
#             # 记录推理结束时间
#             inference_time = time.time() - start_time
#             total_time += inference_time
#
#             target_boxes = extract_boxes_from_targets("voctestceshi.txt")
#             all_ground_truths.extend(target_boxes)
#             # 获取预测框
#             for b in range(batch_size):
#                 # 提取单个样本的预测和目标
#                 # predictions 是一个列表，列表中的每个元素是一个张量（Tensor），代表不同尺度或者不同阶段的预测结果。
#                 # p[b:b+1] 会从张量 p 里提取出第 b 个样本的数据。
#                 # 这里使用 b:b+1 而不是直接用 b，是因为 b:b+1 会保留样本的批次维度，
#                 # 这样提取出来的张量仍然具有批次维度，其形状为 (1, ...)，而 b 提取出来的张量会丢失批次维度。
#                 batch_preds = [p[b:b+1] for p in predictions]
#                 batch_targets = targets[b:b+1]
#
#                 # 从预测中提取框
#                 boxes = extract_boxes_from_predictions(batch_preds, device, config)
#
#                 # 从目标中提取真实框
#                 # target_boxes = extract_boxes_from_targets(batch_targets)
#
#
#                 # 添加到结果列表
#                 all_detections.extend(boxes)
#
#
#     # 计算平均推理时间
#     avg_time_per_image = total_time / max(total_images, 1)
#     fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
#
#     print(f"平均每张图像推理时间: {avg_time_per_image*1000:.2f}ms (FPS: {fps:.2f})")
#
#     # 计算mAP和各类别AP
#     metrics = calculate_map(all_detections, all_ground_truths, config['iou_thresholds'])
#
#     # 添加推理速度指标
#     metrics['overall']['inference_time_ms'] = avg_time_per_image * 1000
#     metrics['overall']['fps'] = fps
#
#     return metrics

def evaluate_model(model, val_loader, device, config=None):
    """
    评估模型性能，计算各种指标

    参数:
        model: 要评估的模型
        val_loader: 验证数据加载器
        device: 设备（'cuda'或'cpu'）
        config: 配置参数

    返回:
        metrics: 评估指标字典
    """

    # 设置模型为评估模式
    model.eval()

    # 收集所有预测结果和真实标注
    all_detections = []
    all_ground_truths = []

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
            targets = targets.to(device)

            # 从文件中读取当前批次对应的目标框
            target_boxes = extract_boxes_from_targets(arg.test_file, start_index, end_index)
            all_ground_truths.extend(target_boxes)
            # 记录推理开始时间
            start_time = time.time()

            # 模型推理
            predictions = model(images)

            # 记录推理结束时间
            inference_time = time.time() - start_time
            total_time += inference_time

            # 获取预测框
            for b in range(batch_size):
                # 提取单个样本的预测和目标
                # predictions 是一个列表，列表中的每个元素是一个张量（Tensor），代表不同尺度或者不同阶段的预测结果。
                # p[b:b+1] 会从张量 p 里提取出第 b 个样本的数据。
                # 这里使用 b:b+1 而不是直接用 b，是因为 b:b+1 会保留样本的批次维度，
                # 这样提取出来的张量仍然具有批次维度，其形状为 (1, ...)，而 b 提取出来的张量会丢失批次维度。
                batch_preds = [p[b:b + 1] for p in predictions]

                # 从预测中提取框
                boxes = extract_boxes_from_predictions(batch_preds, device, config)

                # 添加到结果列表
                all_detections.extend(boxes)
            all_detections = [det.cpu().numpy().tolist() for det in all_detections]
            print("detection: ", all_detections)
            start_index = end_index

    # 计算平均推理时间
    avg_time_per_image = total_time / max(total_images, 1)
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0

    print(f"平均每张图像推理时间: {avg_time_per_image * 1000:.2f}ms (FPS: {fps:.2f})")

    # 计算mAP和各类别AP
    metrics = calculate_map(all_detections, all_ground_truths, config['iou_thresholds'])

    # 添加推理速度指标
    metrics['overall']['inference_time_ms'] = avg_time_per_image * 1000
    metrics['overall']['fps'] = fps

    return metrics


def plot_precision_recall_curve(all_detections, all_ground_truths, class_idx=None, save_path=None):
    """
    绘制精确率-召回率曲线
    
    参数:
        all_detections: 所有检测结果
        all_ground_truths: 所有真实标注
        class_idx: 类别索引，如果为None则绘制所有类别
        save_path: 保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(10, 8))
    
    # 如果指定了类别，只绘制该类别的曲线
    if class_idx is not None:
        class_indices = [class_idx]
    else:
        class_indices = range(CLASS_NUM)
    
    for c in class_indices:
        ap, precision, recall = calculate_ap_per_class(
            all_detections, all_ground_truths, c, 0.5
        )
        
        if len(precision) > 0 and len(recall) > 0:
            plt.plot(recall, precision, label=f'{VOC_CLASSES[c]} (AP={ap:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
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
        result_path = f"eval_results/metrics_{timestamp}.json"
        
        # 将numpy数组转换为列表以便JSON序列化
        json_metrics = metrics.copy()
        for class_name in json_metrics['per_class']:
            for metric in json_metrics['per_class'][class_name]:
                if isinstance(json_metrics['per_class'][class_name][metric], np.float64):
                    json_metrics['per_class'][class_name][metric] = float(json_metrics['per_class'][class_name][metric])
        
        for metric in json_metrics['overall']:
            if isinstance(json_metrics['overall'][metric], np.float64):
                json_metrics['overall'][metric] = float(json_metrics['overall'][metric])
        
        with open(result_path, 'w') as f:
            json.dump(json_metrics, f, indent=4)
        
        print(f"\n评估指标已保存至 {result_path}")
        
        # 绘制PR曲线
        if args['plot_pr_curve']:
            # 收集所有检测和真实标注
            all_detections = []
            all_ground_truths = []
            
            with torch.no_grad():
                for images, targets in tqdm(test_loader, desc="收集PR曲线数据"):
                    batch_size = images.size(0)
                    
                    # 将数据移到指定设备
                    images = images.to(args['device'])
                    targets = targets.to(args['device'])
                    
                    # 模型推理
                    predictions = model(images)
                    
                    # 获取预测框
                    for b in range(batch_size):
                        batch_preds = [p[b:b+1] for p in predictions]
                        batch_targets = targets[b:b+1]
                        
                        boxes = extract_boxes_from_predictions(batch_preds, args['device'], args['config'])
                        target_boxes = extract_boxes_from_targets(batch_targets)
                        
                        all_detections.extend(boxes)
                        all_ground_truths.extend(target_boxes)
            
            # 绘制所有类别的PR曲线
            pr_curve_path = f"eval_results/pr_curve_{timestamp}.png"
            plot_precision_recall_curve(all_detections, all_ground_truths, save_path=pr_curve_path)
    
    return metrics


if __name__ == "__main__":
    main() 
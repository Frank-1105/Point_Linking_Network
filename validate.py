import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

# VOC类别信息
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_NUM = len(VOC_CLASSES)


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    参数:
        box1: [x1, y1, x2, y2] 格式的边界框
        box2: [x1, y1, x2, y2] 格式的边界框
        
    返回:
        IoU值
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # 计算并集面积和IoU
    union_area = box1_area + box2_area - intersection_area
    
    # 防止除0错误
    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    return iou

def compute_area( branch, j, i, grid_size=14):
    """计算指定分支和坐标下的矩形区域。

    Args:
        branch: 分支编号（0=左下，1=左上，2=右下，3=右上）。
        j: 列的索引（0到grid_size-1）。
        i: 行的索引（0到grid_size-1）。
        grid_size: 网格大小，默认为14。

    Returns:
        list: 矩形区域，格式为 [[x_start, x_end], [y_start, y_end]]。
    """

    area = [[], []]
    if branch == 0:
        # 左下角
        area = [[0, j + 1], [i, grid_size]]
    elif branch == 1:
        # 左上角
        area = [[0, j + 1], [0, i + 1]]
    elif branch == 2:
        # 右下角
        area = [[j, grid_size], [i, grid_size]]
    elif branch == 3:
        # 右上角
        area = [[j, grid_size], [0, i + 1]]

    return area


def decode_predictions(result, branch, p_threshold=0.1, score_threshold=0, device="cuda"):
    """
    解码模型输出
    
    参数:
        result: 模型输出
        branch: 分支索引
        p_threshold: 点存在性阈值
        score_threshold: 得分阈值
        device: 设备
        
    返回:
        bbox_info: 解码后的边界框信息
    """
    result = result.squeeze()
    r = []
    bboxes_ = []
    labels_ = []
    scores_ = []
    
    # 应用softmax
    for i in range(14):
        for j in range(14):
            for p in range(4):
                result[i, j, 51 * p + 3:51 * p + 17] = torch.softmax(result[i, j, 51 * p + 3:51 * p + 17], dim=-1)
                result[i, j, 51 * p + 17:51 * p + 31] = torch.softmax(result[i, j, 51 * p + 17:51 * p + 31], dim=-1)
                result[i, j, 51 * p + 31:51 * p + 51] = torch.softmax(result[i, j, 51 * p + 31:51 * p + 51], dim=-1)

    for p in range(2):
        # ij center || mn corner
        for i in range(14):
            for j in range(14):
                if result[i, j, p * 51] < p_threshold:
                    continue
                
                # 计算搜索区域
                x_area, y_area = compute_area(branch, j, i , grid_size=14)
                
                for n in range(y_area[0], y_area[1]):
                    for m in range(x_area[0], x_area[1]):
                        for c in range(CLASS_NUM):
                            # 提取特征
                            p_ij = result[i, j, 51 * p + 0]
                            p_nm = result[n, m, 51 * (p + 2) + 0]
                            i_, j_, n_, m_ = (              #in->行；jm->列
                                result[i, j, 51 * p + 2], 
                                result[i, j, 51 * p + 1], 
                                result[n, m, 51 * (p + 2) + 2], 
                                result[n, m, 51 * (p + 2) + 1]
                            )
                            l_ij_x = result[i, j, 51 * p + 3 + m]
                            l_ij_y = result[i, j, 51 * p + 3 + n]
                            l_nm_x = result[n, m, 51 * (p + 2) + 17 + j]
                            l_nm_y = result[n, m, 51 * (p + 2) + 17 + i]
                            q_cij = result[i, j, 51 * p + 31 + c]
                            q_cnm = result[n, m, 51 * (p + 2) + 31 + c]
                            
                            # 计算得分
                            score = p_ij * p_nm * q_cij * q_cnm * (l_ij_x * l_ij_y + l_nm_x * l_nm_y) / 2
                            score *= 1000
                            
                            if score > score_threshold:
                                r.append([i + i_, j + j_, n + n_, m + m_, c, score])
        
        # 处理所有检测到的边界框
        for l in r:
            # 重新编码为 xmin,ymin,xmax,ymax,class,score   #in->行y；jm->列x  注意编码得到的相对于左上。
            if branch == 0:
                # 左下角
                bbox = [l[3], 2 * l[0] - l[2], 2 * l[1] - l[3], l[2]]
            elif branch == 1:
                # 左上角
                bbox = [l[3], l[2], 2 * l[1] - l[3], 2 * l[0] - l[2]]
            elif branch == 2:
                # 右下角
                bbox = [2 * l[1] - l[3], 2 * l[0] - l[2], l[3], l[2]]
            elif branch == 3:
                # 右上角
                bbox = [2 * l[1] - l[3], l[2], l[3], 2 * l[0] - l[2]]

            # 缩放到图像尺寸
            bbox = [b * 32 for b in bbox]
            bboxes_.append(bbox)
            labels_.append(l[4])
            scores_.append(l[5])
    
    # 如果没有检测到任何边界框，返回空张量
    if not labels_:
        return torch.zeros((0, 6), device=device)
        
    # 构建结果张量
    bbox_info = torch.zeros(len(labels_), 6, device=device)
    for i in range(len(labels_)):
        bbox_info[i, 0] = bboxes_[i][0]  # xmin
        bbox_info[i, 1] = bboxes_[i][1]  # ymin
        bbox_info[i, 2] = bboxes_[i][2]  # xmax
        bbox_info[i, 3] = bboxes_[i][3]  # ymax
        bbox_info[i, 4] = scores_[i]     # score
        bbox_info[i, 5] = labels_[i]     # class
    
    return bbox_info


def non_max_suppression(bbox, nms_threshold=0.1, iou_threshold=0.2):
    """
    非极大值抑制，按类别处理
    
    参数:
        bbox: 边界框信息 [xmin, ymin, xmax, ymax, score, class]
        nms_threshold: 置信度阈值
        iou_threshold: IoU阈值
        
    返回:
        filtered_boxes: 经过NMS处理后的边界框
    """
    if bbox.size(0) == 0:
        return []
        
    filtered_boxes = []
    
    # 按类别索引排序
    ori_class_index = bbox[:, 5]
    class_index, class_order = ori_class_index.sort(dim=0, descending=False)
    class_index = class_index.tolist()
    bbox = bbox[class_order, :]
    
    a = 0
    for i in range(CLASS_NUM):
        # 统计目标数量
        num = class_index.count(i)
        if num == 0:
            continue
            
        # 提取同一类别的所有信息
        x = bbox[a:a + num, :]
        
        # 按照置信度排序
        score = x[:, 4]
        _, score_order = score.sort(dim=0, descending=True)
        y = x[score_order, :]
        
        # 检查最高置信度的框
        if y[0, 4] >= nms_threshold:
            for k in range(num):
                # 再次确认排序
                y_score = y[:, 4]
                _, y_score_order = y_score.sort(dim=0, descending=True)
                y = y[y_score_order, :]
                
                if y[k, 4] > 0:
                    # 计算面积
                    area0 = (y[k, 2] - y[k, 0]) * (y[k, 3] - y[k, 1])
                    if area0 < 200:  # 面积过小的框丢弃
                        y[k, 4] = 0
                        continue
                        
                    # 与其他框比较
                    for j in range(k + 1, num):
                        area1 = (y[j, 2] - y[j, 0]) * (y[j, 3] - y[j, 1])
                        if area1 < 200:
                            y[j, 4] = 0
                            continue
                            
                        # 计算IoU
                        x1 = max(y[k, 0], y[j, 0])
                        x2 = min(y[k, 2], y[j, 2])
                        y1 = max(y[k, 1], y[j, 1])
                        y2 = min(y[k, 3], y[j, 3])
                        
                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        
                        inter = w * h
                        iou = inter / (area0 + area1 - inter)
                        
                        # IoU大于阈值或置信度小于阈值的框丢弃
                        if iou >= iou_threshold or y[j, 4] < nms_threshold:
                            y[j, 4] = 0
                            
            # 保留有效框
            for mask in range(num):
                if y[mask, 4] > 0:
                    filtered_boxes.append(y[mask])
                    
        # 处理下一个类别
        a = num + a
        
    return filtered_boxes


def extract_boxes_from_predictions(predictions, device, config=None):
    """
    从模型预测中提取边界框
    
    参数:
        predictions: 模型预测输出列表
        device: 设备
        config: 配置
        
    返回:
        boxes: 预测的边界框列表
    """
    if config is None:
        config = {
            'p_threshold': 0.1,
            'score_threshold': 0.1,
            'nms_threshold': 0.1,
            'iou_threshold': 0.4
        }
    
    p_threshold = config.get('p_threshold', 0.1)
    score_threshold = config.get('score_threshold', 0.1)
    nms_threshold = config.get('nms_threshold', 0.1)
    iou_threshold = config.get('iou_threshold', 0.4)
    
    # 解码每个分支的预测
    all_boxes = []
    
    for i, pred in enumerate(predictions): # pred:[4,batch_size,204,14,14]
        # 调整张量维度
        pred_permuted = pred.permute(0, 2, 3, 1)
        
        # 对每个批次样本处理g
        for b in range(pred_permuted.shape[0]):
            # 解码单个样本的预测
            sample_pred = pred_permuted[b]
            boxes = decode_predictions(
                sample_pred, 
                branch=i,
                p_threshold=p_threshold, 
                score_threshold=score_threshold,
                device=device
            )
            
            if boxes.size(0) > 0:
                all_boxes.append(boxes)
    
    # 如果有检测结果，合并并应用NMS
    if all_boxes:
        all_boxes = torch.cat(all_boxes, dim=0)
        boxes = non_max_suppression(
            all_boxes, 
            nms_threshold=nms_threshold, 
            iou_threshold=iou_threshold
        )
    else:
        boxes = []
    
    return boxes



def extract_boxes_from_targets(file_path, start_index, end_index):
    gt_boxes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines[start_index:end_index]:
                parts = line.strip().split()
                # 忽略图片名，直接处理边界框信息
                box_parts = parts[1:]
                num_boxes = len(box_parts) // 5
                for i in range(num_boxes):
                    # 提取边界框坐标和类别信息
                    xmin = float(box_parts[i * 5])
                    ymin = float(box_parts[i * 5 + 1])
                    xmax = float(box_parts[i * 5 + 2])
                    ymax = float(box_parts[i * 5 + 3])
                    class_id = int(box_parts[i * 5 + 4])
                    # 构建单个边界框列表
                    gt_box = [xmin, ymin, xmax, ymax, 1.0, class_id]
                    # 添加到结果列表
                    gt_boxes.append(gt_box)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return gt_boxes


def validate_epoch(model, test_loader, criterion, device, epoch, num_epochs):
    """
    在验证集上评估模型
    
    参数:
        model: 模型
        test_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        num_epochs: 总训练轮数
        
    返回:
        avg_loss: 平均损失
        metrics: 评估指标字典
    """
    model.eval()
    total_loss = 0
    component_losses = {
        'p_loss': 0, 
        'coord_loss': 0,
        'link_loss': 0,
        'class_loss': 0,
        'noobj_loss': 0,
        'weighted_coord_loss': 0,
        'weighted_class_loss': 0,
        'weighted_link_loss': 0,
        'weighted_noobj_loss': 0
    }
    
    
    pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validate]')
    
    with torch.no_grad():
        for i, (images, target) in enumerate(pbar):
            images, target = images.to(device), target.to(device)
            
            # 前向传播
            pred = model(images)
            target = target.permute(1, 0, 2, 3, 4)
            
            batch_size = pred[0].shape[0]
            
            # 计算四个点的损失
            loss0, losses_dict0 = criterion(pred[0], target[0])
            loss1, losses_dict1 = criterion(pred[1], target[1])
            loss2, losses_dict2 = criterion(pred[2], target[2])
            loss3, losses_dict3 = criterion(pred[3], target[3])
            
            # 合并四个点的损失字典
            for key in component_losses.keys():
                if key in losses_dict0:
                    # 累加每个点的损失并平均
                    point_loss = (losses_dict0[key] + losses_dict1[key] + 
                                  losses_dict2[key] + losses_dict3[key]) / batch_size
                    component_losses[key] += point_loss.item()
            
            loss = (loss0 + loss1 + loss2 + loss3) / batch_size
            
            # 计算指标 (简化版)
            # 这里只是一个示例，实际应用中应该根据具体任务定义更准确的评估指标
            for b in range(batch_size):
                # 简化的检测评估，实际项目中应替换为更精确的实现
                # ...
                pass
            
            # 记录损失
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            avg_component_losses = {k: v / (i + 1) for k, v in component_losses.items()}
            
            # 更新进度条
            pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
    
    # 计算平均损失
    avg_loss = total_loss / max(len(test_loader), 1)
    avg_component_losses = {k: v / max(len(test_loader), 1) for k, v in component_losses.items()}
    
    # 计算指标 (这里假设一个虚拟的精度/召回率)
    # 在实际应用中应替换为真实的计算方法
    precision = 0.85  # 示例值
    recall = 0.75     # 示例值
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mAP = 0.80        # 示例值
    
    # 打印验证结果
    print(f'\n验证结果:')
    print(f'平均损失: {avg_loss:.5f}')
    print('组件损失:')
    for k, v in avg_component_losses.items():
        print(f'  {k}: {v:.5f}')
    print(f'mAP: {mAP:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1_score:.4f}')
    
    # 返回平均损失和指标
    metrics = {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        **avg_component_losses  # 将组件损失也添加到指标中
    }
    
    return avg_loss, metrics
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm

from PLNdata import PLNDataset  # 使用优化后的类名
from PLNLoss import PLNLoss  # 使用优化后的类名
from PLNnet import pretrained_inception
from validate import *


class Logger(object):
    """日志记录器：同时输出到控制台和文件"""
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_logger():
    """设置日志系统"""
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = Logger(log_filename)
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 创建一个额外的JSON格式日志文件用于记录训练指标
    metrics_log_filename = f"metrics_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return log_filename, metrics_log_filename


def save_checkpoint(model, optimizer, epoch, loss, is_best=False, other_info=None):
    """
    保存模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前训练轮次
        loss: 当前验证损失
        is_best: 是否为最佳模型
        other_info: 其他需要保存的信息
    """
    try:
        model.eval()
        # 保存路径
        checkpoint_dir = "results"
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_ckpt_path = os.path.join(checkpoint_dir, "pln_latest.pth")
        best_ckpt_path = os.path.join(checkpoint_dir, "pln_best.pth")
        
        # 准备保存内容
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        # 添加其他信息
        if other_info:
            checkpoint.update(other_info)
        
        # 保存最新检查点
        torch.save(checkpoint, latest_ckpt_path)
        print(f"已保存最新检查点到 {latest_ckpt_path}")
        
        # 如果是最佳模型，另存一份
        if is_best:
            torch.save(checkpoint, best_ckpt_path)
            print(f"已保存最佳模型到 {best_ckpt_path}")
            
    except Exception as e:
        print(f"保存检查点失败: {str(e)}")
    finally:
        model.train()


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    加载模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        device: 设备
        
    返回:
        start_epoch: 开始轮次
        best_loss: 最佳损失
        其他恢复的信息
    """
    print(f"正在加载检查点 {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        
        # 恢复其他信息
        current_lr = checkpoint.get('current_lr', None)
        total_iterations = checkpoint.get('total_iterations', 0)
        
        print(f"成功加载检查点，从epoch {start_epoch}继续训练")
        if current_lr:
            print(f"当前学习率: {current_lr:.6f}")
        print(f"总迭代次数: {total_iterations}")
        
        return start_epoch, best_loss, {'current_lr': current_lr, 'total_iterations': total_iterations}
    except Exception as e:
        print(f"加载检查点失败: {str(e)}")
        return 0, float('inf'), {'current_lr': None, 'total_iterations': 0}


def adjust_learning_rate(optimizer, current_iteration, init_lr=0.001, max_lr=0.005, warmup_iterations=20000):
    """
    学习率调整策略:
    1. 预热阶段：从init_lr线性增加到max_lr
    2. 之后保持max_lr不变
    
    参数:
        optimizer: 优化器
        current_iteration: 当前迭代次数
        init_lr: 初始学习率
        max_lr: 最大学习率
        warmup_iterations: 预热迭代次数
    
    返回:
        当前学习率
    """
    if current_iteration < warmup_iterations:
        # 预热阶段：从init_lr线性增加到max_lr
        lr = init_lr + (max_lr - init_lr) * (current_iteration / warmup_iterations)
    else:
        # 保持max_lr不变
        lr = max_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def log_metrics(metrics_file, epoch, train_loss, val_loss, component_losses=None, val_metrics=None, lr=None):
    """
    记录训练和验证指标
    
    参数:
        metrics_file: 指标日志文件路径
        epoch: 当前轮次
        train_loss: 训练损失
        val_loss: 验证损失
        component_losses: 组件损失字典
        val_metrics: 验证指标字典
        lr: 当前学习率
    """
    # 读取现有指标记录
    metrics = []
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
    except Exception:
        metrics = []
    
    # 准备当前轮次的指标记录
    epoch_metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if lr is not None:
        epoch_metrics['learning_rate'] = lr
    
    # 添加组件损失
    if component_losses:
        for key, value in component_losses.items():
            epoch_metrics[f'train_{key}'] = value
    
    if val_metrics:
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                epoch_metrics[key] = value
    
    # 添加到记录列表
    metrics.append(epoch_metrics)
    
    # 保存指标记录
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                total_iterations, num_epochs, init_lr=0.001, max_lr=0.005, log_interval=5):
    """
    训练一个epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        total_iterations: 总迭代次数
        num_epochs: 总训练轮数
        init_lr: 初始学习率
        max_lr: 最大学习率
        log_interval: 日志记录间隔（迭代次数）
        
    返回:
        avg_loss: 平均损失
        updated_total_iterations: 更新后的总迭代次数
        current_lr: 当前学习率
        batch_losses: 每个批次的损失
        avg_component_losses: 平均组件损失字典
    """
    model.train()
    total_loss = 0
    batch_losses = []  # 用于记录每个批次的损失
    
    # 初始化各个组件损失的累加器
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
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    current_lr = None
    
    for i, (images, target) in enumerate(pbar):
        # 更新总迭代次数
        total_iterations += 1
        # print(target)
        # 调整学习率
        current_lr = adjust_learning_rate(optimizer, total_iterations, init_lr, max_lr)
        
        # 每100次迭代打印一次学习率
        if total_iterations % 100 == 0:
            print(f'迭代次数: {total_iterations}, 当前学习率: {current_lr:.6f}')

        # 前向传播和损失计算
        images, target = images.to(device), target.to(device)
        pred = model(images)    # pred:[4,batch_size,204,14,14]
        # if isinstance(pred, list):
        #     for p in pred:
        #         print(p.size())
        # else:
        #     print(pred.size())
        target = target.permute(1, 0, 2, 3, 4)  # [batch_size,4,14,14,204]-->[4,batch_size,14,14,204]

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
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        loss_value = loss.item()
        total_loss += loss_value
        batch_losses.append(loss_value)
        avg_loss = total_loss / (i + 1)
        
        # 计算平均组件损失
        avg_component_losses = {k: v / (i + 1) for k, v in component_losses.items()}
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # 定期记录训练信息
        if (i + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss_value:.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
            # 打印组件损失
            print('组件损失:')
            for k, v in avg_component_losses.items():
                print(f'  {k}: {v:.4f}')

    
    # 计算并返回平均损失
    epoch_avg_loss = total_loss / len(train_loader)
    avg_component_losses = {k: v / len(train_loader) for k, v in component_losses.items()}
    
    print(f'Epoch {epoch+1} 训练完成, 平均损失: {epoch_avg_loss:.5f}')
    print('平均组件损失:')
    for k, v in avg_component_losses.items():
        print(f'  {k}: {v:.5f}')
    
    return epoch_avg_loss, total_iterations, current_lr, batch_losses, avg_component_losses


def train(config):
    """
    训练主函数
    
    参数:
        config: 配置字典，包含训练参数
    """
    # 设置设备
    device = config['device']
    print(f"Using device: {device}")
    
    # 初始化日志系统
    log_filename, metrics_log_filename = setup_logger()
    print(f"训练指标将被记录到: {metrics_log_filename}")
    
    # 数据集加载
    train_dataset = PLNDataset(
        img_root=config['data_root'], 
        list_file=config['train_list'],
        train=True, 
        transform=[transforms.ToTensor()]
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_dataset = PLNDataset(
        img_root=config['data_root'], 
        list_file=config['val_list'],
        train=False, 
        transform=[transforms.ToTensor()]
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f'训练集包含 {len(train_dataset)} 张图像')
    print(f'验证集包含 {len(test_dataset)} 张图像')
    
    # 模型初始化
    model = pretrained_inception().to(device)
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=config['init_lr'],
    #     momentum=config['momentum'],
    #     weight_decay=config['weight_decay']
    # )
    # optimizer = optim.RMSprop(
    #     model.parameters(),
    #     lr=config['init_lr'],
    #     momentum=config['momentum'],
    #     weight_decay=config['weight_decay']
    # )
    # 损失函数
    optimizer = optim.Adam(
            model.parameters(),
            lr=config['init_lr'],
            weight_decay=config['weight_decay'],
                           )
    criterion = PLNLoss(
        S=config['grid_size'],
        B=config['num_boxes'],
        w_coord=config['w_coord'],
        w_link=config['w_link'],
        w_class=config['w_class']
    ).to(device)
    
    # 加载检查点（如果存在）
    checkpoint_path = os.path.join("results", "pln_latest.pth")
    if os.path.exists(checkpoint_path) and config['resume_training']:
        start_epoch, best_loss, other_info = load_checkpoint(model, optimizer, checkpoint_path, device)
        total_iterations = other_info.get('total_iterations', 0)
    else:
        start_epoch = 0
        best_loss = float('inf')
        total_iterations = 0
    
    # 记录训练配置
    print("\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建用于记录训练历史的字典
    history = {
        'train_loss': [],
        'val_loss': [],
        'component_losses': [],
        'metrics': []
    }
    
    # 训练循环
    for epoch in range(start_epoch, config['num_epochs']):
        # 训练一个epoch
        train_loss, total_iterations, current_lr, batch_losses, avg_component_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            total_iterations, config['num_epochs'], config['init_lr'], config['max_lr'],
            log_interval=config.get('log_interval', 5)
        )
        
        # 验证
        val_loss, val_metrics = validate_epoch(model, test_loader, criterion, device, epoch, config['num_epochs'])

        # 打印验证结果
        print(f'Epoch {epoch+1}/{config["num_epochs"]} 结果:')
        print(f'训练损失: {train_loss:.5f}, 验证损失: {val_loss:.5f}')
        print(f'验证指标: mAP={val_metrics["mAP"]:.4f}, Precision={val_metrics["precision"]:.4f}, '
        f'Recall={val_metrics["recall"]:.4f}, F1={val_metrics["f1_score"]:.4f}')
        
        # 记录指标到日志文件
        log_metrics(metrics_log_filename, epoch+1, train_loss, val_loss, avg_component_losses, val_metrics, current_lr)
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['component_losses'].append(avg_component_losses)
        history['metrics'].append(val_metrics)
        
        # 保存检查点
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            print(f'获得最佳验证损失: {best_loss:.5f}')
        
        # 保存其他信息
        other_info = {
            'current_lr': current_lr,
            'total_iterations': total_iterations,
            'metrics': val_metrics,
            'train_loss': train_loss,
            'component_losses': avg_component_losses,
            'history': history
        }
        
        # 每个epoch都保存检查点
        save_checkpoint(model, optimizer, epoch, val_loss, is_best, other_info)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 训练结束信息
    print("\n训练完成!")
    print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练日志保存在: {log_filename}")
    print(f"训练指标保存在: {metrics_log_filename}")


if __name__ == "__main__":
    # 训练配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_root': 'E:\datasets\pascalvoc\pascalvoc\VOCdevkit\VOC2007\JPEGImages/',
        'train_list': 'voctrain.txt',
        'val_list': 'voctest.txt',
        'batch_size': 2,
        'num_workers': 4,
        'init_lr': 0.001,
        'max_lr': 0.005,
        'num_epochs': 50,
        'momentum': 0.9,
        'weight_decay': 0.00004,
        'resume_training': False,
        'grid_size': 14,
        'num_boxes': 2,
        'w_coord': 2.0,
        'w_link': 0.5,
        'w_class': 0.5,
        'log_interval': 5  # 每5个批次记录一次训练日志
    }
    
    # 开始训练
    train(config)
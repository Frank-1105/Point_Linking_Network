# # # 初始化类别计数的字典，键为 0 到 19，值初始为 0
# # label_count = {i: 0 for i in range(20)}
# #
# # try:
# #     # 打开文件进行读取操作
# #     with open('voctestceshi.txt', 'r', encoding='utf-8') as file:
# #         # 逐行读取文件内容
# #         for line in file:
# #             # 去除每行首尾的空白字符，并按空格分割成列表
# #             parts = line.strip().split()
# #             # 从第二个元素开始，每 5 个元素一组进行处理
# #             for i in range(1, len(parts), 5):
# #                 try:
# #                     # 提取类别编号
# #                     label = int(parts[i + 4])
# #                     # 检查类别编号是否在 0 到 19 范围内
# #                     if 0 <= label <= 19:
# #                         # 对应类别的计数加 1
# #                         label_count[label] += 1
# #                 except (IndexError, ValueError):
# #                     # 处理可能的索引越界或值错误
# #                     continue
# #
# #     # 打印每个类别的标注数量
# #     for label, count in label_count.items():
# #         print(f"类别 {label} 的标注数为: {count}")
# #
# # except FileNotFoundError:
# #     print("未找到指定的文件，请检查文件路径和文件名。")
# # def extract_boxes_from_targets(file_path):
# #     gt_boxes = []
# #     try:
# #         with open(file_path, 'r', encoding='utf-8') as file:
# #             for line in file:
# #                 parts = line.strip().split()
# #                 # 忽略图片名，直接处理边界框信息
# #                 box_parts = parts[1:]
# #                 num_boxes = len(box_parts) // 5
# #                 for i in range(num_boxes):
# #                     # 提取边界框坐标和类别信息
# #                     xmin = float(box_parts[i * 5])
# #                     ymin = float(box_parts[i * 5 + 1])
# #                     xmax = float(box_parts[i * 5 + 2])
# #                     ymax = float(box_parts[i * 5 + 3])
# #                     class_id = int(box_parts[i * 5 + 4])
# #                     # 构建单个边界框列表
# #                     gt_box = [xmin, ymin, xmax, ymax, 1.0, class_id]
# #                     # 添加到结果列表
# #                     gt_boxes.append(gt_box)
# #     except FileNotFoundError:
# #         print(f"错误：未找到文件 {file_path}。")
# #     except Exception as e:
# #         print(f"发生未知错误：{e}")
# #     return gt_boxes
# #
# # if __name__ == "__main__":
# #     file_path = "voctestceshi.txt"
# #     gt_boxes = extract_boxes_from_targets(file_path)
# #     print(gt_boxes  )
# #     print(len(gt_boxes))
#
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# #
# #
# # def calculate_iou(box1, box2):
# #     """
# #     计算两个边界框的IoU
# #
# #     参数:
# #         box1: [x1, y1, x2, y2] 格式的边界框
# #         box2: [x1, y1, x2, y2] 格式的边界框
# #
# #     返回:
# #         IoU值
# #     """
# #     # 计算交集区域
# #     x1 = max(box1[0], box2[0])
# #     y1 = max(box1[1], box2[1])
# #     x2 = min(box1[2], box2[2])
# #     y2 = min(box1[3], box2[3])
# #
# #     # 计算交集面积
# #     intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
# #
# #     # 计算两个边界框的面积
# #     box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
# #     box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
# #
# #     # 计算并集面积和IoU
# #     union_area = box1_area + box2_area - intersection_area
# #
# #     # 防止除0错误
# #     if union_area == 0:
# #         return 0
# #
# #     iou = intersection_area / union_area
# #     return iou
# #
# #
# # if __name__ == '__main__':
# #     box1 = [162.0, 61.0, 490.0, 307.0]
# #     box2 = [135.0049591064453, 77.73324584960938, 280.11474609375, 361.29498291015625]
# #
# #     iou = calculate_iou(box1, box2)
# #
# #     fig, ax = plt.subplots()
# #
# #     # 绘制第一个边界框
# #     rect1 = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1], linewidth=1, edgecolor='r', facecolor='none')
# #     ax.add_patch(rect1)
# #
# #     # 绘制第二个边界框
# #     rect2 = patches.Rectangle((box2[0], box2[1]), box2[2] - box2[0], box2[3] - box2[1], linewidth=1, edgecolor='b', facecolor='none')
# #     ax.add_patch(rect2)
# #
# #     # 计算交集区域
# #     x1 = max(box1[0], box2[0])
# #     y1 = max(box1[1], box2[1])
# #     x2 = min(box1[2], box2[2])
# #     y2 = min(box1[3], box2[3])
# #
# #     # 绘制交集区域
# #     if x2 > x1 and y2 > y1:
# #         intersection = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='green', alpha=0.3)
# #         ax.add_patch(intersection)
# #
# #     ax.set_xlim(min(box1[0], box2[0]) - 10, max(box1[2], box2[2]) + 10)
# #     ax.set_ylim(min(box1[1], box2[1]) - 10, max(box1[3], box2[3]) + 10)
# #
# #     ax.set_xlabel('X')
# #     ax.set_ylabel('Y')
# #     ax.set_title(f'Intersection over Union (IoU): {iou:.4f}')
# #
# #     plt.show()
# #     print(calculate_iou(box1, box2))
#
#
# import torch
#
# # 假设的类别数量
# CLASS_NUM = 2
#
# def non_max_suppression(bbox, nms_threshold=0.1, iou_threshold=0.2):
#     """
#     非极大值抑制，按类别处理
#
#     参数:
#         bbox: 边界框信息 [xmin, ymin, xmax, ymax, score, class]
#         nms_threshold: 置信度阈值
#         iou_threshold: IoU阈值
#
#     返回:
#         filtered_boxes: 经过NMS处理后的边界框
#     """
#     if bbox.size(0) == 0:
#         return []
#
#     filtered_boxes = []
#
#     # 按类别索引排序
#     ori_class_index = bbox[:, 5]
#     class_index, class_order = ori_class_index.sort(dim=0, descending=False)
#     class_index = class_index.tolist()
#     bbox = bbox[class_order, :]
#
#     a = 0
#     for i in range(CLASS_NUM):
#         # 统计目标数量
#         num = class_index.count(i)
#         if num == 0:
#             continue
#
#         # 提取同一类别的所有信息
#         x = bbox[a:a + num, :]
#
#         # 按照置信度排序
#         score = x[:, 4]
#         _, score_order = score.sort(dim=0, descending=True)
#         y = x[score_order, :]
#
#         # 检查最高置信度的框
#         if y[0, 4] >= nms_threshold:
#             for k in range(num):
#                 # 再次确认排序
#                 y_score = y[:, 4]
#                 _, y_score_order = y_score.sort(dim=0, descending=True)
#                 y = y[y_score_order, :]
#
#                 if y[k, 4] > 0:
#                     # 计算面积
#                     area0 = (y[k, 2] - y[k, 0]) * (y[k, 3] - y[k, 1])
#                     if area0 < 200:  # 面积过小的框丢弃
#                         y[k, 4] = 0
#                         continue
#
#                     # 与其他框比较
#                     for j in range(k + 1, num):
#                         area1 = (y[j, 2] - y[j, 0]) * (y[j, 3] - y[j, 1])
#                         if area1 < 200:
#                             y[j, 4] = 0
#                             continue
#
#                         # 计算IoU
#                         x1 = max(y[k, 0], y[j, 0])
#                         x2 = min(y[k, 2], y[j, 2])
#                         y1 = max(y[k, 1], y[j, 1])
#                         y2 = min(y[k, 3], y[j, 3])
#
#                         w = max(0, x2 - x1)
#                         h = max(0, y2 - y1)
#
#                         inter = w * h
#                         iou = inter / (area0 + area1 - inter)
#
#                         # IoU大于阈值或置信度小于阈值的框丢弃
#                         if iou >= iou_threshold or y[j, 4] < nms_threshold:
#                             y[j, 4] = 0
#
#             # 保留有效框
#             for mask in range(num):
#                 if y[mask, 4] > 0:
#                     filtered_boxes.append(y[mask])
#
#         # 处理下一个类别
#         a = num + a
#
#     return filtered_boxes
#
# if __name__ == '__main__':
#     # 创建一些模拟的边界框数据
#     bbox = torch.tensor([
#         [10, 10, 50, 50, 0.8, 0],  # 类别 0，面积大，置信度高
#         [15, 15, 55, 55, 0.7, 0],  # 类别 0，与上一个框有重叠
#         [200, 200, 210, 210, 0.6, 0],  # 类别 0，面积小
#         [350, 200, 250, 150, 0.9, 1],  # 类别 1，面积大，置信度高
#         [50, 105, 100, 155, 0.8, 1],  # 类别 1，与上一个框有重叠
#         [250, 250, 255, 255, 0.7, 1]  # 类别 1，面积小
#     ])
#
#     # 调用非极大值抑制函数
#     filtered_boxes = non_max_suppression(bbox, nms_threshold=0.1, iou_threshold=0.6)
#
#     # 打印处理后的边界框
#     print("经过NMS处理后的边界框:")
#     for box in filtered_boxes:
#         print(box)
#
#
import torch

if __name__ == "__main__":
    import time
    import numpy as np

    # 测试参数
    GRID_SIZE = 14
    CLASS_NUM = 20  # 假设有20个类别


    # 生成随机测试数据 (模拟模型输出)
    def generate_test_data():
        np.random.seed(42)
        # 生成符合逻辑的测试数据
        data = np.random.rand(GRID_SIZE, GRID_SIZE, 204) * 0.1
        # 确保部分p_ij超过阈值
        data[..., ::51] = np.random.rand(GRID_SIZE, GRID_SIZE, 4) * 0.5 + 0.5  # p_ij在0.5-1.0之间
        # 确保softmax后的值合理
        for p in range(4):
            start = 51 * p + 3
            data[..., start:start + 14] = np.random.dirichlet(np.ones(14), size=(GRID_SIZE, GRID_SIZE))
            data[..., start + 14:start + 28] = np.random.dirichlet(np.ones(14), size=(GRID_SIZE, GRID_SIZE))
            data[..., start + 28:start + 48] = np.random.dirichlet(np.ones(20), size=(GRID_SIZE, GRID_SIZE))
        return torch.from_numpy(data).float()


    # 修正后的decode_predictions函数
    def decode_predictions(result, branch, p_threshold=0.1, score_threshold=0, device="cuda"):
        result = result.to(device).squeeze()

        # 预计算所有需要的索引
        i_idx, j_idx = torch.meshgrid(torch.arange(14, device=device),
                                      torch.arange(14, device=device), indexing='ij')

        # 向量化softmax计算
        for p in range(4):
            start = 51 * p + 3
            result[..., start:start + 14] = torch.softmax(result[..., start:start + 14], dim=-1)
            result[..., start + 14:start + 28] = torch.softmax(result[..., start + 14:start + 28], dim=-1)
            result[..., start + 28:start + 48] = torch.softmax(result[..., start + 28:start + 48], dim=-1)

        # 预计算所有可能组合 (i,j,n,m)
        all_combinations = torch.cartesian_prod(
            torch.arange(14, device=device),
            torch.arange(14, device=device),
            torch.arange(14, device=device),
            torch.arange(14, device=device)
        ).float()

        i, j, n, m = all_combinations.unbind(-1)

        # 计算搜索区域掩码 (简化版)
        valid_mask = torch.ones_like(i, dtype=torch.bool)  # 假设所有区域都有效

        # 提取所有特征 (向量化)
        p_ij = result[i.long(), j.long(), 0::51][:, 0]  # 只取p=0的情况
        p_nm = result[n.long(), m.long(), 102::51][:, 0]  # p=2的情况

        i_ = result[i.long(), j.long(), 2::51][:, 0]
        j_ = result[i.long(), j.long(), 1::51][:, 0]
        n_ = result[n.long(), m.long(), 102 + 2::51][:, 0]
        m_ = result[n.long(), m.long(), 102 + 1::51][:, 0]

        # 使用gather进行向量化查找
        l_ij_x = result[i.long(), j.long(), 3 + m.long()]
        l_ij_y = result[i.long(), j.long(), 3 + n.long()]
        l_nm_x = result[n.long(), m.long(), 102 + 17 + j.long()]
        l_nm_y = result[n.long(), m.long(), 102 + 17 + i.long()]

        # 修正q_scores计算
        q_cij = result[i.long(), j.long(), 31:51]  # [N, 20]
        q_cnm = result[n.long(), m.long(), 102 + 31:102 + 51]  # [N, 20]
        q_scores = q_cij * q_cnm  # [N, 20]

        # 计算总得分
        loc_scores = (l_ij_x * l_ij_y + l_nm_x * l_nm_y) / 2  # [N]
        total_scores = (p_ij * p_nm).unsqueeze(-1) * q_scores * loc_scores.unsqueeze(-1) * 1000  # [N, 20]

        # 应用阈值
        score_mask = (total_scores > score_threshold) & (p_ij.unsqueeze(-1) > p_threshold) & valid_mask.unsqueeze(-1)
        valid_indices = score_mask.nonzero(as_tuple=False)

        if valid_indices.size(0) == 0:
            return torch.zeros((0, 6), device=device)

        # 重组结果
        i_valid = i[valid_indices[:, 0]].float()
        j_valid = j[valid_indices[:, 0]].float()
        n_valid = n[valid_indices[:, 0]].float()
        m_valid = m[valid_indices[:, 0]].float()
        i__valid = i_[valid_indices[:, 0]].float()
        j__valid = j_[valid_indices[:, 0]].float()
        n__valid = n_[valid_indices[:, 0]].float()
        m__valid = m_[valid_indices[:, 0]].float()
        c_valid = valid_indices[:, 1].float()
        scores_valid = total_scores[valid_indices[:, 0], valid_indices[:, 1]]

        # 计算边界框 (向量化)
        if branch == 0:
            bboxes = torch.stack([
                m_valid + m__valid,
                2 * (i_valid + i__valid) - (n_valid + n__valid),
                2 * (j_valid + j__valid) - (m_valid + m__valid),
                n_valid + n__valid
            ], dim=1) * 32
        elif branch == 1:
            bboxes = torch.stack([
                m_valid + m__valid,
                n_valid + n__valid,
                2 * (j_valid + j__valid) - (m_valid + m__valid),
                2 * (i_valid + i__valid) - (n_valid + n__valid)
            ], dim=1) * 32
        elif branch == 2:
            bboxes = torch.stack([
                2 * (j_valid + j__valid) - (m_valid + m__valid),
                2 * (i_valid + i__valid) - (n_valid + n__valid),
                m_valid + m__valid,
                n_valid + n__valid
            ], dim=1) * 32
        elif branch == 3:
            bboxes = torch.stack([
                2 * (j_valid + j__valid) - (m_valid + m__valid),
                n_valid + n__valid,
                m_valid + m__valid,
                2 * (i_valid + i__valid) - (n_valid + n__valid)
            ], dim=1) * 32

        # 组装最终结果
        bbox_info = torch.cat([
            bboxes,
            scores_valid.unsqueeze(-1),
            c_valid.unsqueeze(-1)
        ], dim=1)

        return bbox_info


    # 测试函数
    def test_decode_predictions():
        # 生成测试数据
        test_data = generate_test_data().to("cuda")

        # 测试所有分支
        for branch in range(4):
            print(f"\n测试分支 {branch}...")

            # 计时测试
            start_time = time.time()
            result = decode_predictions(test_data, branch=branch)
            elapsed = time.time() - start_time

            # 打印结果
            print(f"检测到 {len(result)} 个边界框")
            print(f"耗时: {elapsed * 1000:.2f} ms")

            if len(result) > 0:
                print("前5个检测结果:")
                print(result[:5])
            else:
                print("没有检测到任何目标")


    # 运行测试
    print("开始测试 decode_predictions 函数...")
    test_decode_predictions()
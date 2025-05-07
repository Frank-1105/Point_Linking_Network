# from cv2 import cv2
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
# import cv2
from torchvision.transforms import ToTensor

from PLNnet import pretrained_inception, inceptionresnetv2
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
def init_model():
    # model = pretrained_inception().to("cuda")
    model = inceptionresnetv2(num_classes=20, pretrained='imagenet').cuda()
    # # 加载权重，就是在train.py中训练生成的权重文件yolo.pth
    checkpoint = torch.load(r"E:\study\2025spring\baoyan\Paper reproduction\PLN\PLN\results\5\pln_latest.pth", map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    print("Epoch", epoch)
    return model

model = init_model()
# 测试模式
model.eval()
# 设置置信度
score_confident = 0.2
p_confident = 0.3
nms_confident = 0.3
# 设置iou阈值
iou_con = 0.3

# 类别信息，这里写的都是voc数据集的，如果是自己的数据集需要更改
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
# 类别总数：20
CLASS_NUM = len(VOC_CLASSES)

"""
注意：预测和训练的时候是不同的，训练的时候是有标签参考的，在测试的时候会直接输出两个预测框，
保留置信度比较大的，再通过NMS处理得到最终的预测结果，不要跟训练阶段搞混了
"""


class Pred():
    # 参数初始化
    def __init__(self, model, img_root):
        self.model = model
        self.img_root = img_root

    def compute_area(self, branch,j, i):
        area = [[],[]]
        if branch == 0:
            # 左下角
            area = [[0, j+1],[i, 14]]
        elif branch == 1:
            # 左上角
            area = [[0, j+1],[0, i+1]]
        elif branch == 2:
            # 右下角
            area = [[j, 14],[i, 14]]
        elif branch == 3:
            # 右上角
            area = [[j, 14],[0, i+1]]
        #
        # x_area = [[0, a], [0, a],  [a+1, 14],[a+1, 14]]
        # y_area = [[b+1, 14], [0, b], [b+1, 14],[0, b]]
        return area

    def result(self):
        # 读取测试的图像
        img = cv2.imread(self.img_root)
        # 获取高宽信息
        h, w, _ = img.shape
        # 调整图像大小
        image = cv2.resize(img, (448, 448))
        # CV2读取的图像是BGR，这里转回RGB模式
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 图像均值
        mean = (123, 117, 104)  # RGB
        # 减去均值进行标准化操作
        img = img - np.array(mean, dtype=np.float32)
        # 创建数据增强函数
        transform = ToTensor()
        # 图像转为tensor，因为cv2读取的图像是numpy格式
        img = transform(img)
        # 输入要求是BCHW，加一个batch维度
        img = img.unsqueeze(0)
        img = img.to("cuda")
        # print("img",img.shape)
        # 图像输入模型，返回值为1*7*7*204的张量

        Result = self.model(img)
        # Result torch.Size([1, 204, 14, 14])
        # result torch.Size([14, 14, 204])
        # 获取目标的边框信息

        Result0 = Result[0].permute(0, 2, 3, 1).squeeze(0)
        bbox0 = self.Decode(0, Result0)
        Result1 = Result[1].permute(0, 2, 3, 1).squeeze(0)
        bbox1 = self.Decode(1, Result1.clone())
        Result2 = Result[2].permute(0, 2, 3, 1).squeeze(0)
        bbox2 = self.Decode(2, Result2.clone())
        Result3 = Result[3].permute(0, 2, 3, 1).squeeze(0)
        bbox3 = self.Decode(3, Result3)
        print("bbox0", bbox0.shape)
        print("bbox1", bbox1.shape)
        print("bbox2", bbox2.shape)
        print("bbox3", bbox3.shape)
        bbox = torch.cat((bbox0,bbox1,bbox2,bbox3),dim=0)
        print("bbox", bbox.shape)
        print(bbox)
        # 非极大值抑制处理
        bboxes = self.NMS(bbox)  # n*6   bbox坐标是基于7*7网格需要将其转换成448
        if len(bboxes) == 0:
            print("未识别到任何物体")
            print("尝试减小 confident 以及 iou_con")
        for i in range(0, len(bboxes)):  # bbox坐标将其转换为原图像的分辨率
            x1 = bboxes[i][0].item()  # 后面加item()是因为画框时输入的数据不可一味tensor类型
            y1 = bboxes[i][1].item()
            x2 = bboxes[i][2].item()
            y2 = bboxes[i][3].item()
            score = bboxes[i][4].item()
            class_name = bboxes[i][5].item()
            print(x1, y1, x2, y2, VOC_CLASSES[int(class_name)], bboxes[i][4].item())
            text = VOC_CLASSES[int(class_name)] + '{:.3f}'.format(score)
            print("text:", text)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 画框
            cv2.putText(image, text, (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            plt.imsave('test_007.jpg', image)

        # cv2.imwrite("img", image)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

    # 接受的result的形状为1*7*7*204
    def Decode(self, branch, result):
        result = result.squeeze()
        # result [14*14*204]
        # 0 pij 1x 2y
        # 3-16 lxij 16-30 lyij
        # 31-50 qij
        r = []
        bboxes_ = list()
        labels_ = list()
        scores_ = list()
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
                    # print("start,", i, j)
                    if result[i, j, p * 51] < p_confident:
                        continue
                    # print("pij", p, result[i, j, p * 51])
                    # 右上角启发区域
                    # x_area, y_area = [j, 14], [0, i + 1]
                    x_area, y_area = self.compute_area(branch,j,i)
                    for n in range(y_area[0], y_area[1]):
                        for m in range(x_area[0], x_area[1]):
                            for c in range(20):
                                p_ij = result[i, j, 51 * p + 0]
                                p_nm = result[n, m, 51 * (p + 2) + 0]
                                i_, j_, n_, m_ = result[i, j, 51 * p + 2], result[i, j, 51 * p + 1], result[
                                    n, m, 51 * (p + 2) + 2], result[n, m, 51 * (p + 2) + 1]
                                l_ij_x = result[i, j, 51 * p + 3 + m]
                                l_ij_y = result[i, j, 51 * p + 3 + n]
                                l_nm_x = result[n, m, 51 * (p + 2) + 17 + j]
                                l_nm_y = result[n, m, 51 * (p + 2) + 17 + i]
                                q_cij = result[i, j, 51 * p + 31 + c]
                                q_cnm = result[n, m, 51 * (p + 2) + 31 + c]
                                score = p_ij * p_nm * q_cij * q_cnm * (l_ij_x * l_ij_y + l_nm_x * l_nm_y) / 2
                                score *= 1000

                                if score > score_confident:
                                    print(i, j, n, m)
                                    print("score", score)
                                    r.append([i + i_, j + j_, n + n_, m + m_, c, score])
            for l in r:
                # 重新encode 变为xmin,ymin,xmax,ymax,score.class
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

                # print(bbox)
                bbox = [b * 32 for b in bbox]
                bboxes_.append(bbox)
                labels_.append(l[4])
                scores_.append(l[5])  # result of a img
                # bboxes_nms = self._suppress(bboxes_, scores_)
        # print(bboxes_)
        # print(labels_)
        # print(scores_)
        bbox_info = torch.zeros(len(labels_), 6)
        for i in range(len(labels_)):
            bbox_info[i, 0] = bboxes_[i][0]
            bbox_info[i, 1] = bboxes_[i][1]
            bbox_info[i, 2] = bboxes_[i][2]
            bbox_info[i, 3] = bboxes_[i][3]
            bbox_info[i, 4] = scores_[i]
            bbox_info[i, 5] = labels_[i]

        print("bbox_info success")
        print(bbox_info)
        return bbox_info

    # 非极大值抑制处理，按照类别处理，bbox为Decode获取的预测框的位置信息和类别概率和类别信息
    def NMS(self, bbox, iou_con=iou_con):
        # 存放最终需要保留的预测框
        bboxes = []
        # 取出每个gird cell中的类别信息，返回一个列表
        ori_class_index = bbox[:, 5]
        # 按照类别进行排序，从高到低，返回的是排序后的类别列表和对应的索引位置,如下：
        """
        类别排序
        tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
         3.,  3.,  3.,  4.,  4.,  4.,  4.,  5.,  5.,  5.,  6.,  6.,  6.,  6.,
         6.,  6.,  6.,  6.,  7.,  8.,  8.,  8.,  8.,  8., 14., 14., 14., 14.,
        14., 14., 14., 15., 15., 16., 17.], grad_fn=<SortBackward0>)
        位置索引
        tensor([48, 47, 46, 45, 44, 43, 42,  7,  8, 22, 11, 16, 14, 15, 24, 20,  1,  2,
         6,  0, 13, 23, 25, 27, 32, 39, 38, 35, 33, 31, 30, 28,  3, 26, 10, 19,
         9, 12, 29, 41, 40, 21, 37, 36, 34, 18, 17,  5,  4])
        """
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        # class_index是一个tensor，这里把他转为列表形式
        class_index = class_index.tolist()
        # 根据排序后的索引更改bbox排列顺序
        bbox = bbox[class_order, :]
        a = 0
        for i in range(0, CLASS_NUM):
            # 统计目标数量，即某个类别出现在grid cell中的次数
            num = class_index.count(i)
            # 预测框中没有这个类别就直接跳过
            if num == 0:
                continue
            # 提取同一类别的所有信息
            x = bbox[a:a + num, :]
            # 提取真实类别概率信息
            score = x[:, 4]
            # 提取出来的某一类别按照真实类别概率信息高度排序，递减
            score_index, score_order = score.sort(dim=0, descending=True)
            # 根据排序后的结果更改真实类别的概率排布
            y = x[score_order, :]
            # 先看排在第一位的物体的概率是否大有给定的阈值，不满足就不看这个类别了，丢弃全部的预测框
            if y[0, 4] >= nms_confident:
                for k in range(0, num):
                    # 真实类别概率，排序后的
                    y_score = y[:, 4]
                    # 对真实类别概率重新排序，保证排列顺序依照递减，其实跟上面一样的，多此一举
                    _, y_score_order = y_score.sort(dim=0, descending=True)
                    y = y[y_score_order, :]
                    # 判断概率是否大于0
                    if y[k, 4] > 0:
                        # 计算预测框的面积
                        area0 = (y[k, 2] - y[k, 0]) * (y[k, 3] - y[k, 1])
                        if area0 < 200:
                            y[k, 4] = 0
                            continue
                        for j in range(k + 1, num):
                            # 计算剩余的预测框的面积
                            area1 = (y[j, 2] - y[j, 0]) * (y[j, 3] - y[j, 1])
                            if area1 < 200:
                                y[j, 4] = 0
                                continue
                            x1 = max(y[k, 0], y[j, 0])
                            x2 = min(y[k, 2], y[j, 2])
                            y1 = max(y[k, 1], y[j, 1])
                            y2 = min(y[k, 3], y[j, 3])
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0 or h < 0:
                                w = 0
                                h = 0
                            inter = w * h
                            # 计算与真实目标概率最大的那个框的iou
                            iou = inter / (area0 + area1 - inter)
                            # iou大于一定值则认为两个bbox识别了同一物体删除置信度较小的bbox
                            # 同时物体类别概率小于一定值也认为不包含物体
                            if iou >= iou_con or y[j, 4] < nms_confident:
                                y[j, 4] = 0
                for mask in range(0, num):
                    if y[mask, 4] > 0:
                        bboxes.append(y[mask])
            # 进入下个类别
            a = num + a
        #     返回最终预测的框
        return bboxes


if __name__ == "__main__":
    Pred = Pred(model, img_root)
    Pred.result()

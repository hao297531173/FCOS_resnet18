import torch
from torch import nn
import torch.nn.functional as f
import torchvision.models as models
import numpy as np
import loss
"""
这个文件是网络的架构
"""





# 以resnet18作为backbone的FCOS
class FCOS_18(nn.Module):

    # 构造函数
    def __init__(self,p_stage_num=5, class_num=80,  MODEL_ROOT="./model/resnet18-5c106cde.pth", feature=256):
        super().__init__()
        self.p_stage_num = p_stage_num
        self.class_num = class_num
        """
        加载resnet18网络作为backbone
        """
        pretrained_net = models.resnet18(pretrained=False)
        pre = torch.load(MODEL_ROOT)
        pretrained_net.load_state_dict(pre)
        self.backbone = pretrained_net

        """
        backbone的特征提取部分
        stage_c3的通道数为128
        stage_c4输出的通道数为256
        stage_c5输出的通道数为512
        
        输出如下
        torch.Size([1, 3, 224, 224])
        torch.Size([1, 128, 28, 28])    8倍下采样
        torch.Size([1, 256, 14, 14])    16倍下采样
        torch.Size([1, 512, 7, 7])      32倍下采样
        
        torch.Size([1, 3, 800, 1024])
        torch.Size([1, 128, 100, 128])
        torch.Size([1, 256, 50, 64])
        torch.Size([1, 512, 25, 32])
        """
        self.stage_c3 = nn.Sequential(*list(self.backbone.children())[:-4])
        self.stage_c4 = list(self.backbone.children())[-4]
        self.stage_C5 = list(self.backbone.children())[-3]

        """
        下面定义从c->p的卷积层，主要用于变换通道数，p的通道数都为feature
        
        输入出如下
        torch.Size([1, 3, 224, 224])
        torch.Size([1, 256, 28, 28])
        torch.Size([1, 256, 14, 14])
        torch.Size([1, 256, 7, 7])
        
        torch.Size([1, 3, 800, 1024])
        torch.Size([1, 256, 100, 128])
        torch.Size([1, 256, 50, 64])
        torch.Size([1, 256, 25, 32])
        """
        self.conv_c3 = nn.Conv2d(128, feature, kernel_size=1)
        self.conv_c4 = nn.Conv2d(256, feature, kernel_size=1)
        self.conv_c5 = nn.Conv2d(512, feature, kernel_size=1)

        """
        下面是p阶段的上下采样
        
        [p3,p4,p5,p6,p7]的输出如下
        torch.Size([1, 3, 800, 1024])
        torch.Size([1, 256, 100, 128])
        torch.Size([1, 256, 50, 64])
        torch.Size([1, 256, 25, 32])
        torch.Size([1, 256, 13, 16])
        torch.Size([1, 256, 7, 8])
        """
        self.stage_p6 = nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1)
        self.stage_p7 = nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1)



        """
        下面是head里面用到的卷积层
        """
        self.head_cls = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1)
        )
        self.head_reg = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, stride=1)
        )
        # 用于得到分类信息
        self.head_cls_cls = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1, stride=1)
        # 用于得到centerness
        self.head_cls_cen = nn.Conv2d(feature, 1, kernel_size=3, padding=1, stride=1)
        # 用于得到regression
        self.head_reg_reg = nn.Conv2d(feature, 4, kernel_size=3, padding=1, stride=1)


        # 用来存储结果
        self.outcome = None

        # 记录每个层的形状
        self.stage_p = []

    # 上采样，mode参数默认的是"nearest",使用mode="bilinear"的时候会有warning
    def upsampling(self, src, target, mode="nearest"):
        # target的形状举例 torch.Size([1, 256, 50, 64])
        return f.interpolate(src, size=[target.shape[2], target.shape[3]], mode=mode)


    # 根据索引返回各个batch_size的三个预测结果
    # self.outcome的维数是[batch_size, 5, 3]
    # 这个输出结果还没解决
    """
    标准用法
    print(fcos[i][0].shape)
    print(fcos[i][1].shape)
    print(fcos[i][2].shape)
    """
    def __getitem__(self, index):
        # assert(len(self.outcome)==5), "是不是没有前向传播呀，我这里维数不对呀"
        return self.outcome

    def __len__(self):
        return self.p_stage_num

    def getp3(self):
        return self.outcome[0]

    def getp4(self):
        return self.outcome[1]

    def getp5(self):
        return self.outcome[2]

    def getp6(self):
        return self.outcome[3]

    def getp7(self):
        return self.outcome[4]

    # 返回[batchsize, 5, h*w, c]
    def getClassification(self, index):
        return self.outcome[index][0]

    def getCenterness(self, index):
        return self.outcome[index][1]

    def getRegression(self, index):
        return self.outcome[index][2]

    # 将所有的结果存在一个列表中返回
    # output = [cls, cen, reg]
    # cls = [p3, p4, p5, p6, p7] 的分类向量
    # cen = [p3, p4, p5, p6, p7] 的centerness
    # reg = [p3, p4, p5, p6, p7] 的回归向量
    def getAll(self):
        cls = []
        cen = []
        reg = []
        for i in range(5):
            cls.append(self.getClassification(i))
            cen.append(self.getCenterness(i))
            reg.append(self.getRegression(i))
        output = []
        output.append(cls)
        output.append(cen)
        output.append(reg)
        return output

    # 这段代码用来得到[p3,p4,p5,p6,p7]
    def c_p_stage_forward(self, input):
        c3 = self.stage_c3(input)
        c4 = self.stage_c4(c3)
        c5 = self.stage_C5(c4)
        p5 = self.conv_c5(c5)
        p4 = self.conv_c4(c4)
        p3 = self.conv_c3(c3)
        p6 = self.stage_p6(p5)
        p7 = self.stage_p7(p6)

        p4 = self.upsampling(p5, p4)+p4
        p3 = self.upsampling(p4, p3)+p3
        """
        print(p3.shape)
        print(p4.shape)
        print(p5.shape)
        print(p6.shape)
        print(p7.shape)
        """
        return [p3,p4,p5,p6,p7]

    # 这段代码用来得到每一层的[classification, centerness, regression]
    def head(self, input):
        # print("input->"+str(input.shape))
        cls = self.head_cls(input)
        reg = self.head_reg(input)
        cls_cls = self.head_cls_cls(cls)
        cls_cen = self.head_cls_cen(cls)
        reg_reg = self.head_reg_reg(reg)
        
        # 将回归参数经过exp函数
        # 我就简单一点，直接使用exp函数
        # 将输出值映射到(0, ∞)
        reg_reg = torch.exp(reg_reg)
        cls_cls = torch.exp(cls_cls)
        cls_cen = torch.exp(cls_cen)
        return [cls_cls, cls_cen, reg_reg]

    # 用于返回每一层的最终计算结果，输出维度为[5, 3]
    def output(self, input):
        assert (len(input)==5), "输入维数不为5，别慌，不是大问题"
        output = []
        for i in range(len(input)):
            output.append(self.head(input[i]))
        return output


    def forward(self, input, reshape=False):
        # 计算p3-p7
        stage_p = self.c_p_stage_forward(input)
        # 计算最后的预测结果
        output = self.output(stage_p)
        self.outcome = output

        self.outcome_origin = self.outcome

        if(reshape == True):
            # 将形状改变

            for i in range(len(self.outcome)):
                for j in range(len(self.outcome[0])):
                    self.outcome[i][j] = np.transpose(self.outcome[i][j].detach().numpy(), (0,2,3,1))
                    self.outcome[i][j] = torch.from_numpy(self.outcome[i][j])
                    # 对特征图进行合并，也就是将二维的向量变成一维向量
                    batch_size = self.outcome[i][j].shape[0]
                    w, h = self.outcome[i][j].shape[1], self.outcome[i][j].shape[2]
                    m = self.outcome[i][j].shape[3]
                    self.outcome[i][j] = self.outcome[i][j].reshape(batch_size, w*h, m)
        return output


# 这个类用来生成结果进行模型评估
class FCOS_eval(nn.Module):

    # score_threshold 预测分数的阈值
    # nms_iou_threshold nms算法的iou阈值
    # max_detection_boxes_num 最多取的预测框数量
    # 默认的分数是0.05， nms的iou阈值是0.6， 最多选1000个候选框
    def __init__(self,model_path,score_threshold=0.05,nms_iou_threshold=0.6,max_detection_boxes_num=1000,gpu=1):
        super().__init__()
        # 这个用来返回计算结果
        # [cls_logits, cnt_logits, reg_preds]
        self.fcos = FCOS_18()
        self.model_path = model_path
        # 加载训练好的模型
        #self.fcos.load_state_dict(torch.load(self.model_path))
        print(self.model_path + " 模型加载完毕✿✿ヽ(°▽°)ノ✿")
        self.fcos_output = None
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes = max_detection_boxes_num
        self.strides = [8,16,32,64,128]
        self.loss = loss.LossFunction()
        if(gpu==1):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.fcos = self.fcos.to(self.device)


    # [batch_size, n, h, w] -> [batch_size, sum(h*w), n]
    # coords -> [sum(h*w), 2]
    def reshape(self, input):
        batch_size = input[0].shape[0]
        n = input[0].shape[1]
        out = []
        coords = []
        for i in range(5):
            input[i] = input[i].permute(0,2,3,1)
            coord = self.loss.feature2input(input[i], self.strides[i])
            input[i] = torch.reshape(input[i], [batch_size, -1, n])
            out.append(input[i])
            coords.append(coord)
        out = torch.cat(out, dim=1)
        coords = torch.cat(coords, dim=0)
        return out, coords


    # 根据坐标信息得到候选框
    # coords -> [batch_size, sum(h*w)]
    # reg_preds -> [batch_size, sum(h*w), 4] ltrb
    # 这个函数比较好理解，就是得到候选框的左上角和右下角坐标
    # boxes -> [batch_size, sum(h*w), [x1,y1,x2,y2]]
    def getBoxes(self, coords, reg_preds):
        # [batch_size, sum(h*w), 2]
        x1y1 = coords[None, :, :].to(self.device) - reg_preds[..., :2].to(self.device)
        x2y2 = coords[None, :, :].to(self.device) + reg_preds[..., 2:].to(self.device)
        # [batch_size, sum(h*w), 4]
        boxes = torch.cat([x1y1, x2y2], dim=-1)
        return boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)


    # 进行nms非极大值抑制
    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep


    # 对候选框进行后处理
    # preds -> [cls_score_topk, cls_class_topk, box_topk]
    def post_process(self, preds):
        _cls_score_post = []
        _cls_class_post = []
        _box_post = []
        cls_score_topk, cls_class_topk, box_topk = preds
        for batch in range(cls_score_topk.shape[0]):
            # 筛选到分数小于阈值的框
            mask = cls_score_topk[batch] >= self.score_threshold
            cls_score_b = cls_score_topk[batch][mask]
            cls_class_b = cls_class_topk[batch][mask]
            box_b = box_topk[batch][mask]
            # 非极大值抑制算法
            nms_ind = self.batched_nms(box_b, cls_score_b, cls_class_b, self.nms_iou_threshold)
            _cls_score_post.append(cls_score_b[nms_ind])
            _cls_class_post.append(cls_class_b[nms_ind])
            _box_post.append(box_b[nms_ind])

        scores = torch.stack(_cls_score_post, dim=0)
        classes = torch.stack(_cls_class_post, dim=0)
        boxes = torch.stack(_box_post, dim=0)
        return scores, classes, boxes

    # 选出预测框
    # input的形状和self.fcos_output的形状一样
    def detection(self, input):
        # 首先将input reshape
        cls_logits, cnt_logits, reg_logits = input
        cls_logits, coords = self.reshape(cls_logits)
        cnt_logits, _ = self.reshape(cnt_logits)
        reg_preds, _ = self.reshape(reg_logits)

        # 将cls_logits和cnt_logits经过sigmoid函数
        cls_logits = cls_logits.sigmoid_()
        cnt_logits = cnt_logits.sigmoid_()

        coords = coords.to(self.device)

        # 得到每个点的分类类别和得分
        # [batch_size, sum(h*w)]
        cls_score, cls_class = torch.max(cls_logits, dim=-1)
        cls_score = cls_score.to(self.device)
        cls_class = cls_class.to(self.device)
        # 乘上centerness然后开根号
        cls_score = torch.sqrt(cls_score*(cnt_logits.squeeze(dim=-1))).to(self.device)
        # 类别标签在[1,80]之间
        cls_class = cls_class+1

        # 得到候选框
        # boxes -> [batch_size, sum(h * w), [x1, y1, x2, y2]]
        boxes = self.getBoxes(coords, reg_preds)
        boxes = boxes.to(self.device)

        # 获得topk个候选框
        # 首先看一下最大框数和预测出来的框哪个小，取小的那个
        num_k = min(self.max_detection_boxes, cls_score.shape[-1])
        # [batch_size, num_k]
        _, top_k_ind = torch.topk(cls_score, num_k, dim=-1, largest=True, sorted=True)
        # 将topk信息记录下来
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_score.shape[0]):
            # [num_k]
            _cls_scores.append(cls_score[batch][top_k_ind[batch]])
            _cls_classes.append(cls_class[batch][top_k_ind[batch]])
            # [num_k, 4]
            _boxes.append(boxes[batch][top_k_ind[batch]])
        # [batch_size, num_k]
        cls_score_topk = torch.stack(_cls_scores, dim=0)
        cls_class_topk = torch.stack(_cls_classes, dim=0)
        # [batch_size, num_k, 4]
        boxes_topk = torch.stack(_boxes, dim=0)
        preds = [cls_score_topk, cls_class_topk, boxes_topk]
        scores, classes, boxes = self.post_process(preds)
        return scores, classes, boxes

    # 对box进行剪裁
    def clipBoxes(self, batch_img, batch_box):
        batch_box = batch_box.clamp_(min=0)
        h, w = batch_img.shape[2:]
        batch_box[..., [0, 2]] = batch_box[..., [0, 2]].clamp_(max=w - 1)
        batch_box[..., [1, 3]] = batch_box[..., [1, 3]].clamp_(max=h - 1)
        return batch_box








    def forward(self, input):
        self.fcos(input)
        # [cls_logits, cnt_logits, reg_logits]
        self.fcos_output = self.fcos.getAll()
        scores, classes, boxes = self.detection(self.fcos_output)
        boxes = self.clipBoxes(input, boxes)
        return scores, classes, boxes









if __name__ == "__main__":
    fcos = FCOS_18()
    fcos_e = FCOS_eval("./model/fcos_res18.pth", gpu=1)
    input = torch.rand(2,3,800,1024).to(fcos_e.device)
    output = fcos_e(input)
    print(output)
    """
    fcos(input)
    for i in range(5):
        print(fcos.getClassification(i).shape)
        print(fcos.getCenterness(i).shape)
        print(fcos.getRegression(i).shape)
    
    output = fcos.getAll()
    cls, cen, reg = output
    for i in range(5):
        print(cls[i].shape)
        print(cen[i].shape)
        print(reg[i].shape)
    """



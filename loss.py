import torch
import torch.nn as nn
import torch.nn.functional as f
import network
import numpy as np
import dataload
"""
这个脚本用来计算损失函数
"""

class LossFunction(nn.Module):

    # strides是一个列表，里面存放着每一层的下采样倍数
    # limit_range也是一个列表，里面存放着每一层检测目标的大小范围
    def __init__(self,strides=None, limit_range=None, add_centerness=True, gpu=1):
        super().__init__()
        if(gpu==1):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu()")
        else:
            self.device = torch.device("cpu")
        if(strides==None):
            self.strides = [8,16,32,64,128]
        else:
            self.strides = strides

        if(limit_range==None):
            self.limit_range = [[0,64],[64,128],[128,256],[256,512],[512,9999999999]]
        else:
            self.limit_range = limit_range
        self.add_centerness = add_centerness

        warning = "输出的层数信息是不是不对呀，下采样有"+str(len(self.strides))+"层，但是限制信息有"+str(len(self.limit_range))+"层"
        assert (len(self.strides)==len(self.limit_range))

    # 将特征图上的点映射到输入图像中的坐标
    # (x,y) --> (s/2+sx, s/2+sy)
    # 需要注意的是在送到这个函数之前，向量已经改变形状了
    # feature = [batch_sizel, h, w, :]
    def feature2input(self, feature, stride):
        # 先得到feature的宽和高
        w, h = feature.shape[2], feature.shape[1]
        # 计算x和y映射到输入图像的坐标
        x = torch.arange(0, w*stride, stride, dtype=torch.float32)
        y = torch.arange(0, h*stride, stride, dtype=torch.float32)

        # 将x和y变成w*h的形状
        x, y = torch.meshgrid(x, y)

        # 将x和y拉成一维的形状
        x = torch.reshape(x, [-1])
        y = torch.reshape(y, [-1])

        # 计算最后的值
        coords = torch.stack([x, y], dim=-1) + stride//2
        return coords

    # 计算每一层的mask
    # return
    # cls_target, cnt_target, reg_target
    def cal_level_targets(self, level_out, gt_box, gt_class, stride, limit_range
                          ,sample_radiu_ratio=1.5):
        # 分别取出三个特征图
        cls_feature, cen_feature, reg_feature = level_out
        # 获取batch_size数
        batch_size = len(cls_feature)
        # 获取分类类别数
        class_num = cls_feature.shape[1]
        # 获取标签中gt_box的数量
        gt_box_num = gt_box.shape[1]

        # 将特征图分别做映射
        cls_feature = cls_feature.permute(0,2,3,1)  # [batch_size, h, w, class_num]
        # 计算坐标值
        coords = self.feature2input(cls_feature, stride)

        cls_feature = cls_feature.reshape((batch_size, -1, class_num)) # [batch_size, h*w, class_num]
        cnt_feature = cen_feature.permute(0,2,3,1)  # [batch_size, h. w. 1]
        cnt_feature = cen_feature.reshape((batch_size, -1, 1))
        reg_feature = reg_feature.permute(0,2,3,1)
        reg_feature = reg_feature.reshape((batch_size, -1, 4))

        # 特征图的面积大小
        feature_area = cls_feature.shape[1]

        # 计算特征图上每个点对于gt_box的(l,t,r,b)值
        x = coords[:, 0]
        y = coords[:, 1]
        # [1, h*w, 1] - [batch_size, 1, m] = [batch_size, h*w, m]
        # 其中None表示增加一个维度
        l_off = x[None, :, None] - gt_box[..., 0][:, None, :].cpu()
        t_off = y[None, :, None] - gt_box[..., 1][:, None, :].cpu()
        r_off = gt_box[..., 2][:, None, :].cpu() - x[None, :, None]
        b_off = gt_box[..., 3][:, None, :].cpu() - y[None, :, None]

        # 将上面四个张量聚合在一起，dim=-1代表在后面增加一个维度
        # [batch_size, h*w, m, 4]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)

        # 计算每一个gt_box的面积
        # [batch_size, h*w, m]
        areas = (ltrb_off[..., 0]+ltrb_off[..., 2])*(ltrb_off[..., 1]+ltrb_off[..., 3])

        # 在每一组(l,t,r,b)中选出最大值和最小值，索引0表示值，索引1表示最小(大)值下标
        # [batch_size, h*w, m]
        off_min = torch.min(ltrb_off, dim=-1)[0]
        off_max = torch.max(ltrb_off, dim=-1)[0]

        """
        下面是一些判断条件
        """
        # 取最小值大于0的索引
        mask_in_gtbox = off_min > 0

        # print("mask_in_gtbox->" + str((mask_in_gtbox==0).all()))

        # 判断gt_box的大小是否在该层检测范围内
        mask_in_level = (off_max>limit_range[0])&(off_max<limit_range[1])

        # 判断特征图映射的点是否离gt_box中心太远
        radiu = stride*sample_radiu_ratio
        gt_x_center = (gt_box[...,0]+gt_box[...,2])/2
        gt_y_center = (gt_box[...,1]+gt_box[...,3])/2
        # 计算映射点和gt_box中心点的ltrb，然后找到最大值，筛选到最大值小于radiu的点
        # [1, h*w, 1] - [batch_size, 1, m] = [batch_size, h*w, m]
        # 这四个值里面有正有负，之后我们取最大值就可以得到正的那个最大值了
        c_l_off = x[None, :, None] - gt_x_center[:, None, :].cpu()
        c_t_off = y[None, :, None] - gt_y_center[:, None, :].cpu()
        c_r_off = gt_x_center[:, None, :].cpu() - x[None, :, None]
        c_b_off = gt_y_center[:, None, :].cpu() - y[None, :, None]

        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]

        # 这里逻辑不要搞错了，筛选掉远离中心点的值
        mask_center = c_off_max < radiu


        # mask_pos就是最终的结果
        mask_pos = mask_in_gtbox & mask_in_level & mask_center
        """
        print("mask_pos->" + str(mask_pos.shape))
        print("mask_center->" + str(mask_center.shape))
        print("mask_in_level->" + str(mask_in_gtbox.shape))
        print("mask_in_gtbox->" + str(mask_in_gtbox.shape))
        print("coords->" + str(coords.shape))
        """

        # ~表示取反，就是将0的区域设置一个非常大的值
        areas[~mask_pos] = 999999999

        """
        之前我们计算过areas了，它的维度是[batch_size, h*w, m]
        也就是说每个特征图上的点都有m个gt_box的面积
        我们需要取对于每个点来首最小的那个gt_box的ltrb作为reg_target
        """
        # [batch_size, h*w]
        area_min_index = torch.min(areas, dim=-1)[1]
        reg_index = torch.zeros_like(areas, dtype=torch.uint8).scatter_(-1, area_min_index.unsqueeze(-1),1)
        # reg_target记录的就是特征图上每个点的ltrb值
        reg_target = ltrb_off[reg_index]
        # [batch_size, h*w, 4]
        reg_target = torch.reshape(reg_target, (batch_size, -1, 4))

        # 将classes的维度变得和areas一样
        # [batch_size, h*w, m]
        classes = torch.broadcast_tensors(gt_class[:,None,:], areas.long())[0]
        cls_target = classes[torch.zeros_like(areas, dtype=torch.uint8).scatter_(-1, area_min_index.unsqueeze(-1),1)]
        # [batch_size, h*w, 1]
        cls_target = torch.reshape(cls_target, (batch_size, -1, 1))

        # 下面计算centerness
        # [batch_size, h*w]
        left_right_min = torch.min(reg_target[...,0], reg_target[...,2])
        left_right_max = torch.max(reg_target[...,0], reg_target[...,2])
        top_bottom_min = torch.min(reg_target[...,1], reg_target[...,3])
        top_bottom_max = torch.max(reg_target[...,1], reg_target[...,3])
        # [batch_size, h*w, 1]
        cnt_target = ((left_right_min/left_right_max)*(top_bottom_min/top_bottom_max)).sqrt()
        cnt_target = cnt_target.unsqueeze(-1)
        
        """
        print(cnt_target.shape)
        print(batch_size)
        print(feature_area)
        """

        warning = "target的维数好像不对"
        assert (reg_target.shape==(batch_size, feature_area, 4)), "reg_"+warning
        assert (cls_target.shape==(batch_size, feature_area, 1)), "cls_"+warning
        assert (cnt_target.shape==(batch_size, feature_area, 1)), "cnt_"+warning
        

        # 计算target里面不取1的地方
        # [batch_size, h*w, m] -> [batch_size, h*w]
        mask_pos_2 = mask_pos.long().sum(dim=-1)
        # 大于等于1表示这个特征图上的点有框预测
        mask_pos_2 = mask_pos_2 >= 1
        warning = "mask_pos_2的维数不对"
        # assert (mask_pos_2==(batch_size, feature_area)), warning
        cls_target[~mask_pos_2] = 0
        cnt_target[~mask_pos_2] = -1
        reg_target[~mask_pos_2] = -1

        return cls_target, cnt_target, reg_target


    # input_network是网络得到的结果 5*[cls, cen, reg]
    # gt_box, gt_class是coco数据集的标签值
    # 得到的结果是mask矩阵
    def getTargets(self, input_network, gt_box, gt_class):
        # 从输入中取出三个值
        # cls_out = [5, [batch, 80, w, h]]
        # cen_out = [5, [batch, 1, w, h]]
        # reg_out = [5, [batch, 4, w, h]]
        cls_out, cen_out, reg_out = input_network

        # 用于存放结果
        cls_targets = []
        cen_targets = []
        reg_targets = []

        # 检查是否将所有p层的结果都放进来了
        warning = "下采样层数好像不对呀，输入结果只有"+str(len(cls_out))+"层，但是strides里面有"+str(len(self.strides))+"层"
        assert (len(self.strides)==len(cls_out)), warning

        # 对于每一层都计算损失值
        for level in range(len(cls_out)):
            level_out = [cls_out[level], cen_out[level], reg_out[level]]
            level_targets = self.cal_level_targets(level_out, gt_box, gt_class, self.strides[level],
                                                self.limit_range[level])

            cls_targets.append(level_targets[0])
            cen_targets.append(level_targets[1])
            reg_targets.append(level_targets[2])

        # 将结果压缩成很多行，并且返回
        # 返回格式是 分类损失，centerness损失和回归损失
        cls_targets_tensor = torch.cat(cls_targets, dim=1)
        cen_targets_tensor = torch.cat(cen_targets, dim=1)
        reg_targets_tensor = torch.cat(reg_targets, dim=1)
        return cls_targets_tensor, cen_targets_tensor, reg_targets_tensor



    """
    下面计算损失函数
    """
    # 计算分类损失
    # preds = [5, [batch_size, class_num, h, w]] list
    # targets = [batch_size, h*w, 1]
    # mask = [batch_size, h*w]
    def compute_cls_loss(self, preds, targets, mask):
        batch_size = targets.shape[0]
        preds_reshape = []
        class_num = preds[0].shape[1]
        # mask = [batch_size, h*w, 1]
        mask = mask.unsqueeze(dim=-1)
        # 计算每个batch需要计算的结点个数
        num_pos = torch.sum(mask, dim=[1,2]).clamp_(min=1).float()

        #######################
        # 打印需要计算的结点个数
        ######################
        # print("需要计算结点个数: "+str(int(num_pos[0].item())))
        # 将preds改变形状->[batch_size, h*w, class_num]
        for pred in preds:
            pred = pred.permute(0,2,3,1)
            pred = torch.reshape(pred, [batch_size, -1, class_num])
            preds_reshape.append(pred)
        # [batch_size, sum(h*w), class_num]
        preds = torch.cat(preds_reshape, dim=1)

        warning = "compute_cls_loss-->reds的维数和targets的维数不一样"
        assert (preds.shape[:2]==targets.shape[:2]), warning


        # 筛选出需要计算的点
        pred_list = []
        target_list = []
        pred_batch = []
        target_batch = []
        for batch_size in range(len(mask)):
            for idx in range(len(mask[0])):
                if(mask[0][idx][0]>0):
                    pred_list.append(preds[batch_size][idx].unsqueeze(dim=0))
                    # print("pred[batch_size][idx].shape="+str(preds[batch_size][idx].unsqueeze(dim=0).shape))
                    target_list.append(targets[batch_size][idx].unsqueeze(dim=0))
            # [sum(h*w), class_num]
            pred_batch.append(torch.cat(pred_list, dim=0).unsqueeze(dim=0))
            # [sum(h*w), 1]
            target_batch.append(torch.cat(target_list, dim=0).unsqueeze(dim=0))
        pred_batch = torch.cat(pred_batch, dim=0)
        target_batch = torch.cat(target_batch, dim=0)
        # print("pred_batch="+str(pred_batch.shape))
        # print("target_batch="+str(target_batch.shape))

        # 计算损失值
        loss = []
        for batch_index in range(len(pred_batch)):
            pred_pos = pred_batch[batch_index]  # [sum(_h*_w),class_num]
            # print(pred_pos.shape)
            target_pos = target_batch[batch_index]  # [sum(_h*_w),1]

            # print(target_pos.shape)
            # print("target_pos.shape->"+str(target_pos.shape))
            # print("target_pos->" + str(targets[batch_index][targets[batch_index]>0]))
            # target_pos = [sum(h*w), class_num]也就是说变成了sum(h*w)行的onehot
            target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None,:]==target_pos).float()

            # print(torch.sum(target_pos>0))
            loss.append(self.focal_loss(pred_pos, target_pos).view(1))

        return torch.cat(loss, dim=0).to(self.device) / num_pos.to(self.device)  # [batch_size,]


    # 计算centerness损失
    # preds = [5, [batch_size, 1, h, w]] list
    # targets = [batch_size, h*w, 1]
    # mask = [batch_size, h*w]
    def compute_cnt_loss(self, preds, targets, mask):
        batch_size = targets.shape[0]
        preds_reshape = []
        c = targets.shape[-1]
        mask = mask.unsqueeze(dim=-1)
        # print((mask==0).all())
        # [batch_size, ]
        num_pos = torch.sum(mask, dim=[1,2]).clamp_(min=1).float()
        # print("compute_cnt_loss->num_pos="+str(num_pos))
        for pred in preds:
            pred = pred.permute(0,2,3,1)
            pred = torch.reshape(pred, [batch_size, -1, c])
            preds_reshape.append(pred)

        preds = torch.cat(preds_reshape, dim=1)
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index][mask[batch_index]]
            target_pos = targets[batch_index][mask[batch_index]]
            # print("mask[batch_index]->num " + str(torch.sum(mask[batch_index]>0)))
            # print("pred_pos->shape "+str(pred_pos.shape))
            loss_item = f.binary_cross_entropy_with_logits(pred_pos.cpu(), target_pos.cpu(), reduction="sum").view(1)
            loss.append(loss_item)
        # print("cnt_loss-->" + str(torch.cat(loss, dim=0)))
        return torch.cat(loss, dim=0).to(self.device)/num_pos.to(self.device)   # [batch_size, ]


    # 计算iou损失函数
    # pred = [n, 4]   ltrb
    # target = [n, 4]
    def iou_loss(self, pred, target):
        # print(pred.shape)
        # print(target.shape)
        # 首先要计算交集面积 area=(t+r)*(t+b)
        # 选出ltrb中小的一组值
        lt = torch.min(pred[:, :2].to(self.device), target[:, :2].to(self.device))
        rb = torch.min(pred[:, 2:].to(self.device), target[:, 2:].to(self.device))
        wh = (lt+rb).clamp(min=0)
        overlap = (wh[:, 0]*wh[:, 1]).to(self.device)
        
        # 分别计算预测框和gt_box的面积
        area1 = ((pred[:, 0]+pred[:, 2])*(pred[:, 1]+pred[:, 3])).to(self.device)
        area2 = ((target[:, 0]+target[:, 2])*(target[:, 1]+target[:, 3])).to(self.device)
        iou = overlap/(area1+area2-overlap)
        # 类似交叉熵损失
        iou_loss = -iou.clamp(min=1e-6).log()
        #print(iou_loss.shape)
        #print("iou_loss-->" + str(iou_loss.sum()))
        return iou_loss.sum()

    # 不懂这个函数为什么这么算
    def giou_loss(self, pred, target):
        lt_min = torch.min(pred[:, :2], target[:, :2])
        rb_min = torch.min(pred[:, 2:], target[:, 2:])
        wh_min = (lt_min+rb_min).clamp(min=0)
        overlap = wh_min[:, 0]*wh_min[:, 1]
        area1 = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
        area2 = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
        union = area1 + area2 - overlap
        iou = overlap / union

        lt_max = torch.max(pred[:, :2], target[:, :2])
        rb_max = torch.max(pred[:, 2:], target[:, 2:])
        wh_max = (lt_max+rb_max).clamp(min=0)
        G_area = wh_max[:, 0]*wh_max[:, 1]
        giou = iou - (G_area-union)/G_area.clamp(min = 1e-10)
        loss = 1.-giou
        # print("giou_loss-->" + str(loss))
        return loss.sum()





    # 计算回归损失
    # preds = [5, [batch_size, 1, h, w]] list
    # targets = [batch_size, h*w, 4]
    # mask = [batch_size, h*w]
    def compute_reg_loss(self, preds, targets, mask, mode="iou"):
        batch_size = targets.shape[0]
        c = targets.shape[-1]
        preds_reshape = []
        num_pos = torch.sum(mask, dim=[1]).clamp_(min=1).float()
        # print(num_pos)
        for pred in preds:
            pred = pred.permute(0,2,3,1)
            pred = torch.reshape(pred, [batch_size, -1, c])
            preds_reshape.append(pred)

        preds = torch.cat(preds_reshape, dim=1)
        # print("preds->shape " + str(preds.shape))
        loss = []
        for batch_index in range(batch_size):
            # [sum(h*w), 4]
            # print(mask[batch_index])
            pred_pos = preds[batch_index][mask[batch_index]]
            # print("mask[batch_index]->num " + str(torch.sum(mask[batch_index]>0)))
            # print("pred_pos->shape "+str(pred_pos.shape))
            target_pos = targets[batch_index][mask[batch_index]]
            if mode == 'iou':
                loss.append(self.iou_loss(pred_pos, target_pos).view(1))
            elif mode == 'giou':
                loss.append(self.giou_loss(pred_pos, target_pos).view(1))
            else:
                raise NotImplementedError("reg loss only implemented ['iou','giou']")
        return torch.cat(loss, dim=0).to(self.device) / num_pos.to(self.device)  # [batch_size,]






    # Focal loss
    # pred [n, class_num]
    # target [n, class_num]，每一行只有一个值为1

    """
    这里需要注意，计算的时候有可能标签值很小，导致损失值趋近于无穷大
    """
    def focal_loss(self, preds, targets, gamma=2, alpha=0.25):
        m = nn.Softmax(dim=1)
        preds = m(preds).to(self.device)
        # 进行clamp，防止出现nan
        preds = preds.clamp(min=0.0001, max=0.9999)
        targets = targets.to(self.device)
        pt = preds*targets + (1.0-preds)*(1.0-targets)
        # pt = m(pt)
        # print("pt->" + str(pt))
        w = alpha*targets + (1.0-alpha)*(1.0-targets)
        # print(w)
        loss = -w*torch.pow((1-pt), gamma)*pt.log()
        # print("focal_loss-->" + str(loss.sum()))
        return loss.sum()



    # 损失函数计算入口
    """
    inputs_pred是network进行计算后 getAll()返回的结果 
    [5, [[batch, class_num, ...]], [[batch, 1, ...]], [[batch, 4, ...]]]
    inputs_targets是标签值
    [[batch, bbox_num, 4], [batch, bbox_num]]    
    """
    def forward(self, inputs_pred, inputs_targets, print_info=False):
        cls_logits, cnt_logits, reg_preds = inputs_pred
        targets_ = self.getTargets(inputs_pred, inputs_targets[0], inputs_targets[1])
        cls_targets, cnt_targets, reg_targets = targets_
        # 这里三个mask应该是共享的，cnt_targets>-1那么就代表这个点需要计算
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)  # [batch_size,sum(_h*_w)]
        # print("forward->num "+str(torch.sum(mask_pos>0)))
        # print(cnt_targets)
        # 函数的返回值应该是[batch_size]，然后对他们做平均
        cls_loss = self.compute_cls_loss(cls_logits, cls_targets, mask_pos).mean()
        cnt_loss = self.compute_cnt_loss(cnt_logits, cnt_targets, mask_pos).mean().clamp(max=1.0)
        reg_loss = self.compute_reg_loss(reg_preds, reg_targets, mask_pos).mean()


        if(print_info == True):
            """
            cls_loss-->tensor([178098.9062, 178088.7188], grad_fn=<DivBackward0>)
            cnt_loss-->tensor([11854.7461, 11941.8975], grad_fn=<DivBackward0>)
            reg_loss-->tensor([235746.9688, 235746.9688], grad_fn=<DivBackward0>)
            """

            print("cls_loss-->" + str(cls_loss))
            print("cnt_loss-->" + str(cnt_loss))
            print("reg_loss-->" + str(reg_loss))

        if self.add_centerness:
            total_loss = cls_loss + cnt_loss + reg_loss
            return cls_loss, cnt_loss, reg_loss, total_loss
        else:
            total_loss = cls_loss + reg_loss + cnt_loss * 0.0
            return cls_loss, cnt_loss, reg_loss, total_loss




if __name__ == "__main__":
    # 获得网络的输出值
    num = 1
    function = LossFunction(gpu=0)
    fcos = network.FCOS_18().to(function.device)
    dataset = dataload.COCOdataset(is_train=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=num, shuffle=True,
                                               collate_fn=dataset.collate_fn)
    input = None
    target_bbox = None
    target_cls = None
    for data in train_loader:
        input = data[0].to(function.device)
        target_bbox = data[1].to(function.device)
        target_cls = data[2].to(function.device)
        break
    # print(target_bbox.shape)
    # print(target_cls)

    fcos(input)
    output_input = fcos.getAll()
    """
    for i in range(len(output_input)):
        for j in range(len(output_input[i])):
            print((output_input[i][j]>0).all())
    """
    cls, cen, reg = output_input
    
    losses = function(output_input, [target_bbox, target_cls], print_info=True)
    cls_loss, cnt_loss, reg_loss, total_loss = losses
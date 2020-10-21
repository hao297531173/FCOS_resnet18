import torch
from torch import nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as tfs
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import dataloader
import numpy as np
from torchvision.datasets import CocoDetection
import cv2


class COCOdataset(CocoDetection):


    # 类别名称
    CLASSES_NAME = (
    '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush')

    prefix = "G:/工作空间/文献/语义分割/图像分割论文/语义分割论文/FCOS/FCOS-PyTorch-37.2AP/data/coco/"

    def __init__(self, img_path=None, ann_path=None, resize_size=[800,1333],is_train = True, transform=None):

        self.resize_size = resize_size
        self.transform = transform
        self.is_train = is_train
        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]

        # 标签与下标的对应关系
        self.category2index = {}
        self.index2category = {}
        # 根据标签找到类名
        self.index2name = {}

        self.img_path = img_path
        self.ann_path = ann_path

        # 如果构造函数没有填写图片地址和标注地址，那么就是用默认地址
        if(img_path==None and ann_path==None):
            if(is_train==False):
                self.img_path = self.prefix+"val2017/"
                self.ann_path = self.prefix+"annotations/instances_val2017.json"
            else:
                self.img_path = self.prefix + "train2017/"
                self.ann_path = self.prefix + "annotations/instances_train2017.json"

        # 调用基类的构造函数，加载标注文件
        super().__init__(self.img_path, self.ann_path)

        print("加载文件成功，一共加载了："+str(len(self.ids)) + "张图片")

        self.filter()

        # 加载标签与下标对应关系
        self.initcat()

        print("标签与下标对应关系加载完成...")
        print("COCO数据加载器初始化完成✿✿ヽ(°▽°)ノ✿")





    # 对加载的图片进行过滤
    def filter(self):
        ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            # 如果标签是空的，那么就跳过这个标签
            if(len(ann)==0):
                pass


            # 一个ann表示的是一张图片中的对象，一张图片有很多被标注的对象
            # 每个对象都有一个bbox，这些bbox里面只要有一个bbox为空（长或宽小于等于1）
            # 那么就舍弃这个标注
            abandon = False     # 循环结束时加入这个元素为True，那么就舍弃这个标注
            obj_num = 0
            for obj in ann:
                obj_num += 1
                w, h = obj["bbox"][2], obj["bbox"][3]
                if(w<=1 or h<=1):
                    abandon = True
                    break
            if(abandon == True):
                pass
            elif(obj_num == 0):
                pass
            else:
                ids.append(id)
        self.ids = ids
        print("过滤后，剩下的图片数量为：" + str(len(self.ids)))


    # 获取标签与下标的对应关系
    def initcat(self):
        for index, cat in enumerate(self.coco.getCatIds()):
            self.category2index[cat] = index+1

        for key, value in self.category2index.items():
            self.index2category[value] = key

        for cat in self.coco.getCatIds():
            self.index2name[cat] = self.CLASSES_NAME[self.category2index[cat]]



    # 通过index得到标签，其中index是一个列表
    def getCatName(self, index):
        assert (len(index)!=0), "列表好像是空的呀，快检查一下代码哪里有错"
        name = []
        for i in range(len(index)):
            # classname = self.CLASSES_NAME[self.category2index[index[i]]]
            name.append(self.index2name[index[i]])
        return name

    # 将图片和bbox进行缩放
    def processImg(self, image, bbox, input_ksize=None, has_bbox=True):
        if(input_ksize==None):
            input_ksize = self.resize_size
        h, w = image.shape[:2]

        min_size, max_size = input_ksize

        smallest_side = min(h, w)
        largest_side = max(h, w)

        scale = min_size/smallest_side
        if(largest_side*scale > max_size):
            scale = max_size/largest_side

        nw, nh = int(scale*w), int(scale*h)
        image_resize = cv2.resize(image, (nw, nh))

        # 这里是填充的数量，让nw和nh可以除得尽32
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        # 将image_resized变成np的类型
        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        # 相当于(nh, nw)之外的部分填充0，右边和下边
        image_paded[:nh, :nw, :] = image_resize

        if bbox is None:
            return image_paded
        else:
            # 将bbox也放缩scale倍数
            bbox[:, [0,2]] = bbox[:, [0,2]] * scale
            bbox[:, [1,3]] = bbox[:, [1,3]] * scale
            return image_paded, bbox




    # 重载__getitem__方法
    def __getitem__(self, index):
        img, ann = super().__getitem__(index)

        # 过滤掉"iscrowd"不为0的属性
        ann = [o for o in ann if(o["iscrowd"]==0)]
        # 获得bbox
        bbox = [o["bbox"] for o in ann]
        bbox = np.array(bbox, dtype=np.float32)

        # 将bbox的坐标由 xywh的性质转变为x1y1x2y2
        bbox[..., 2:] = bbox[..., 2:] + bbox[..., :2]

        if(self.is_train==True and self.transform!=None):
            img, bbox = self.transform(img, bbox)

        img = np.array(img)

        img, bbox = self.processImg(img, bbox)

        # 类别信息
        classes = [o['category_id'] for o in ann]
        classes = [self.category2index[c] for c in classes]
        classes = np.array(classes)

        img = tfs.ToTensor()(img)
        bbox = torch.from_numpy(bbox)
        classes = torch.LongTensor(classes)

        return img, bbox, classes

    # 将一个批次的所有图片和bbox填充成相同的形状
    # 这个函数作为torch.utils.data.DataLoader的一个参数传递
    def collate_fn(self, data):
        img_list, bbox_list, class_list = zip(*data)

        assert(len(img_list)==len(bbox_list)==len(class_list)), "形状好像不太一样啊，快检查一下代码"

        # 获取batch的大小
        batch_size = len(img_list)
        # 用于存放填充后的图像，bbox和类别信息
        pad_img = []
        pad_bbox = []
        pad_class = []


        """
        开始填充图片
        """
        # 找到图像集中最大的长和宽
        h_list = [int(img.shape[1]) for img in img_list]
        w_list = [int(img.shape[2]) for img in img_list]

        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        # 对所有图像在右边和下边进行填充
        for i in range(len(img_list)):
            img = img_list[i]
            # 第二个参数是填充的行数或者列数
            # 分别是左右上下
            img = f.pad(img, [0, int(max_w-img.shape[2]), 0, int(max_h-img.shape[1])], value=0.)
            img = tfs.Normalize(self.mean, self.std, inplace=True)(img)
            pad_img.append(img)

        """
        开始填充bbox
        """
        # 找到一个batch中一张图片中的最多的bbox的数量
        max_num = 0
        for i in range(batch_size):
            n = bbox_list[i].shape[0]
            if(n > max_num):
                max_num = n

        # 对bbox向量进行下填充, 对类别向量进行右填充
        for i in range(batch_size):
            box = bbox_list[i]
            # box可能是一个空列表
            box = f.pad(box, [0,0,0,max_num-box.shape[0]], value=-1)
            pad_bbox.append(box)
            classid = class_list[i]
            classid = f.pad(classid, [0, max_num-classid.shape[0]], value=-1)
            pad_class.append(classid)

        # 将三个列表变成tensor并返回
        # 使用torch.stack()将数据组织成[batch, ...]的格式
        # torch.stack()里面的参数可以是一个装有tensor的列表
        pad_img = torch.stack(pad_img)
        pad_bbox = torch.stack(pad_bbox)
        pad_class = torch.stack(pad_class)

        return pad_img, pad_bbox, pad_class


if __name__ == "__main__":
    dataset = COCOdataset(is_train=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                               collate_fn=dataset.collate_fn)

    # data = [图片，gt_box，类别标签]
    """
    torch.Size([8, 3, 1184, 1216])
    torch.Size([8, 15, 4])
    torch.Size([8, 15])
    """
    for data in train_loader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[2])
        break

    print("跑通啦\(^o^)/~")



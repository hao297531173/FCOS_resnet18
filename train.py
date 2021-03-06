import torch
from torch.utils.data import DataLoader
import numpy as np
import network
import loss
import dataload
import argparse
import time
import matplotlib.pyplot as plt
import os

"""
超参数设定
"""
# 测试
EPOCHES = 24
BATCH_SIZE = 8
LINUX = 1
LR = 0.001
GPU = 1
RESTART = 0
STEP = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=EPOCHES)
parser.add_argument("--batch", type=int, default=BATCH_SIZE)
parser.add_argument("--linux", type=int, default=LINUX)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--gpu", type=int, default=GPU)
# 不读取checkpoint，重新开始训练
parser.add_argument("--restart", type=int, default=RESTART)
# 设置多少步打印一次
parser.add_argument("--step", type=int, default=STEP)
opt = parser.parse_args()

# 设定路径
if(opt.linux==0):
    pre = "G:/工作空间/文献/语义分割/图像分割论文/语义分割论文/FCOS/FCOS-PyTorch-37.2AP/"
    root_train = pre+"data/coco/train2017"
    root_train_ann = pre+"data/coco/annotations/instances_train2017.json"
    root_val = pre+"data/coco/val2017"
    root_val_ann = pre+"data/coco/annotations/instances_val2017.json"
else:
    """
    待填
    """
    root_train = "/home/yyl/data/coco/train2017"
    root_train_ann = "/home/yyl/data/coco/annotations/instances_train2017.json"
    root_val = "/home/yyl/data/coco/val2017"
    root_val_ann = "/home/yyl/data/coco/annotations/instances_val2017.json"

# 所有训练的函数都写在这个类里面
class Boot(object):

    def __init__(self, model_path="./fcos_res18.pth"):
        super().__init__()
        self.model_path = model_path
        self.EPOCHES = opt.epoch

        # 用于记录损失函数
        self.total_loss = []
        self.cls_loss = []
        self.cnt_loss = []
        self.reg_loss = []

        # 用于记录验证集的损失函数
        self.val_total_loss = []
        self.val_cls_loss = []
        self.val_cnt_loss = []
        self.val_reg_loss = []

        self.train_dataset = dataload.COCOdataset(img_path=root_train, ann_path=root_train_ann)
        self.val_dataset = dataload.COCOdataset(img_path=root_val, ann_path=root_val_ann)
        # 加载成批量数据
        self.train_data = DataLoader(self.train_dataset, batch_size=opt.batch, shuffle=True,
                                           collate_fn=self.train_dataset.collate_fn)
        self.val_data = DataLoader(self.val_dataset, batch_size=opt.batch, shuffle=True,
                                           collate_fn=self.val_dataset.collate_fn)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu()")
        # 加载模型
        self.net = network.FCOS_18()
        # 加载优化器
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
        # 表示当前训练的轮次偏移量
        self.offset = 0
        # 加载损失函数
        if(opt.gpu==1):
            self.lossFunction = loss.LossFunction(gpu=1)
        else:
            self.lossFunction = loss.LossFunction(gpu=0)
        # 如果有GPU就是用GPU
        if(opt.gpu == 1):
            self.net = self.net.to(self.device)
            self.lossFunction = self.lossFunction.to(self.device)

        if(opt.restart == 1):
            with open("checkpoint.txt", "w") as f:
                f.write("epoch=-1")

        if os.path.exists("./checkpoint"):

            print("读取检查点信息")
            filename = os.listdir("./checkpoint")
            
            with open("checkpoint.txt", "r") as f:
                lines = f.readlines()
                self.offset = int(lines[0].split("=")[-1])
                    

            if(self.offset==-1):
                self.offset = 0
                
                # 删除checkpoint的文件
                if not os.path.exists("./checkpoint"):
                    os.mkdir("./checkpoint")
                dirs = os.listdir("./checkpoint")


                if (len(dirs)!=0):  # 如果文件夹不空
                    # 删除之前保存的模型
                    for dir in dirs:
                        if(dir.split(".")[-1] == "pth"):
                            os.remove("./checkpoint/"+dir)

                dirs = os.listdir("./")
                for dir in dirs:
                    if(dir == "checkpoint.txt"):
                        os.remove("./checkpoint.txt")
            else:  # 加载模型，继续训练
                self.net.load_state_dict(torch.load("./checkpoint/"+filename[0]))
                print("加载模型完毕，模型路径为："+"./checkpoint/"+filename[0])
        else:
            os.mkdir("./checkpoint")
        print("读取checkpoint信息，得到上次训练的轮次为第%d轮"%(self.offset))
        print("训练器初始化完成，✿✿ヽ(°▽°)ノ✿")

    # 返回每轮的训练信息
    def getinfo(self, epoch, total_loss, cls_loss, cnt_loss, reg_loss, cost_time):
        info  = "epoch %d: total_loss: %.4f cls_loss: %.4f, cnt_loss: %.4f, reg_loss: %.4f"%\
                (epoch, total_loss, cls_loss, cnt_loss, reg_loss)
        info = info + " cost time: %dh: %dm: %ds"%\
               (cost_time/3600, cost_time%3600//60, cost_time%60)
        return info

    # 绘图
    # prefix是保存图片的地址前缀
    def draw(self, prefix="./image/", is_val=False):
        if not os.path.exists(prefix):
            os.mkdir(prefix)

        x = np.arange(0, self.EPOCHES)
        plt.plot(x, self.total_loss, label="train")
        if(is_val==True):
            plt.plot(x, self.val_total_loss, label="val")
        plt.title("total_loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("total loss")
        plt.savefig(prefix+"total_loss.jpg")
        plt.clf()

        plt.plot(x, self.cls_loss, label="train")
        if (is_val == True):
            plt.plot(x, self.val_cls_loss, label="val")
        plt.title("cls_loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("cls loss")
        plt.savefig(prefix + "cls_loss.jpg")
        plt.clf()

        plt.plot(x, self.cnt_loss, label="train")
        if (is_val == True):
            plt.plot(x, self.val_cnt_loss, label="val")
        plt.title("cnt_loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("cnt loss")
        plt.savefig(prefix + "cnt_loss.jpg")
        plt.clf()

        plt.plot(x, self.reg_loss, label="train")
        if (is_val == True):
            plt.plot(x, self.val_reg_loss, label="val")
        plt.title("reg_loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("reg loss")
        plt.savefig(prefix + "reg_loss.jpg")
        plt.clf()

        # 再画两个个总的图吧
        plt.plot(x, self.total_loss, label="total loss")
        plt.plot(x, self.cls_loss, label="cls loss")
        plt.plot(x, self.cnt_loss, label="cnt loss")
        plt.plot(x, self.reg_loss, label="reg loss")
        plt.title("losses")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(prefix + "losses_train.jpg")
        plt.clf()

        if(is_val==True):
            plt.plot(x, self.val_total_loss, label="total loss")
            plt.plot(x, self.val_cls_loss, label="cls loss")
            plt.plot(x, self.val_cnt_loss, label="cnt loss")
            plt.plot(x, self.val_reg_loss, label="reg loss")
            plt.title("losses")
            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig(prefix + "losses_val.jpg")
            plt.clf()

    # 保存模型
    def save_model(self, prefix="./model/"):
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        torch.save(self.net.state_dict(), prefix+"fcos_res18.pth")


    # 每一轮epoch训练完都保存一下模型
    def checkpoint(self, epoch, prefix="./checkpoint/"):
        # 检查指定文件夹是否存在，不存在则创建
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        dirs = os.listdir(prefix)


        if (len(dirs)!=0):  # 如果文件夹不空
            # 删除之前保存的模型
            for dir in dirs:
                if(dir.split(".")[-1] == "pth"):
                    os.remove(prefix+dir)


        # 将epoch写入txt文件
        with open("checkpoint.txt", "w") as f:
            f.write("epoch=" + str(epoch+self.offset))

        # 保存新模型
        save_path = prefix + "checkpoint_"+str(epoch+self.offset) + ".pth"
        torch.save(self.net.state_dict(), save_path)
        print("epoch %d:保存检查点成功，ヾ(◍°∇°◍)ﾉﾞ"%(epoch))

    # 对验证集进行测试
    def val(self, epoch, test):
        _total_loss = 0
        _cls_loss = 0
        _cnt_loss = 0
        _reg_loss = 0
        num_loss = 0
        # 开始计时
        start_time = time.time()
        for step, data in enumerate(self.val_data):
            batch_img, batch_bbox, batch_classes = data
            if (opt.gpu == 1):
                batch_img = batch_img.to(self.device)
                batch_bbox = batch_bbox.to(self.device)
                batch_classes = batch_classes.to(self.device)

            # 前向传播
            self.net(batch_img)
            # 获取运算结果
            output = self.net.getAll()
            # 计算损失函数
            losses = self.lossFunction(output, [batch_bbox, batch_classes])
            # 记录损失函数值
            _total_loss += losses[-1]
            _cls_loss += losses[0]
            _cnt_loss += losses[1]
            _reg_loss += losses[2]
            num_loss += 1

            if(test==1):
                break

        # 将每一轮的损失值保存
        self.val_total_loss.append(_total_loss / num_loss)
        self.val_cls_loss.append(_cls_loss / num_loss)
        self.val_cnt_loss.append(_cnt_loss / num_loss)
        self.val_reg_loss.append(_reg_loss / num_loss)

        end_time = time.time()
        cost_time = int(end_time - start_time)
        print("val->"+self.getinfo(epoch, _total_loss / num_loss, _cls_loss / num_loss,
                           _cnt_loss / num_loss, _reg_loss / num_loss, cost_time))

    # offset表示前面已经训练了多少轮
    # 当offset=-1的时候就表示是刚刚开始训练
    def train(self, is_val=False, test=0):
        print("开始训练，ヾ(◍°∇°◍)ﾉﾞ，从第%d轮开始，一共训练%d轮"%(self.offset, self.EPOCHES))
        epoches = self.EPOCHES-self.offset
        for epoch in range(epoches):
            _total_loss = 0
            _cls_loss = 0
            _cnt_loss = 0
            _reg_loss = 0
            num_loss = 0
            # 开始计时
            start_time = time.time()
            for step, data in enumerate(self.train_data):
                batch_img, batch_bbox, batch_classes = data
                if(opt.gpu == 1):
                    batch_img = batch_img.to(self.device)
                    batch_bbox = batch_bbox.to(self.device)
                    batch_classes = batch_classes.to(self.device)

                # 前向传播
                self.net(batch_img)
                # 获取运算结果
                output = self.net.getAll()

                """
                # 检查输出值是否有小于0的值
                for i in range(len(output)):
                    for j in range(len(output[i])):
                        if((output[i][j]>0).all().item()!=1):
                            output[i][j].clamp(min=0.00001, max=1)
                            if((output[i][j]==0).all().item()!=1 and (output[i][j]==0).all().item()!=1):
                                print("网络输出存在小于等于0的值，是不是哪里有点问题呀")
                            elif((output[i][j]==0).all().item()!=1):
                                print("网络输出存在等于0的值，是不是哪里有点问题呀")
                            elif((output[i][j]==0).all().item()!=1):
                                print("网络输出存在小于等于0的值，是不是哪里有点问题呀")
                """
                # 计算损失函数
                losses = self.lossFunction(output, [batch_bbox, batch_classes])
                # 记录损失函数值
                _total_loss += losses[-1]
                _cls_loss += losses[0]
                _cnt_loss += losses[1]
                _reg_loss += losses[2]
                loss = losses[-1].mean()
                # 反向传播，更新梯度
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                num_loss += 1
                end_time = time.time()
                cost_time = int(end_time-start_time)
                #print("step%d ->"%(step)+self.getinfo(epoch+self.offset, _total_loss/num_loss, _cls_loss/num_loss,\
                #              _cnt_loss/num_loss, _reg_loss/num_loss, cost_time))

                if(step%opt.step==0 and step!=0):
                    print("step%d ->"%(step)+self.getinfo(epoch+self.offset, _total_loss/num_loss, _cls_loss/num_loss,
                               _cnt_loss/num_loss, _reg_loss/num_loss, cost_time))

                if(test == 1):
                    break
            # 将每一轮的损失值保存
            self.total_loss.append(_total_loss/num_loss)
            self.cls_loss.append(_cls_loss/num_loss)
            self.cnt_loss.append(_cnt_loss/num_loss)
            self.reg_loss.append(_reg_loss/num_loss)

            end_time = time.time()
            cost_time = int(end_time-start_time)
            print("train->"+self.getinfo(epoch+self.offset, _total_loss/num_loss, _cls_loss/num_loss,
                               _cnt_loss/num_loss, _reg_loss/num_loss, cost_time))
            self.checkpoint(epoch)
            if(is_val==True):
                self.val(epoch, test)

        print("训练完成✿✿ヽ(°▽°)ノ✿，开始绘图和保存模型")
        # 训练完成后就绘图
        self.draw()
        # 保存模型
        self.save_model()
        print("绘图完成，保存模型完成✿✿ヽ(°▽°)ノ✿")

    # 开始函数
    def start(self, is_val=False, test=0):
        if(opt.restart==1):
            self.checkpoint(-1)
        self.train(is_val, test)






if __name__ == "__main__":
    train_boot = Boot()
    train_boot.start(is_val=False, test=0)
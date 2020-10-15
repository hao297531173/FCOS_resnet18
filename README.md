# FCOS_resnet18
 FCOS代码的复现，backbone使用的是resnet18

这段代码是复现fcos的代码，backbone使用了resnet18
主要分为两个部分

第一部分训练网络，入口函数是train.py

运行脚本的时候可以设置外部参数

--epoch  设置训练轮数
--batch  设置batch_size
--linux  等于1的时候代表读取linux上的数据集，等于0代表读取win10上的
--lr     设置学习率，默认为0.01
--gpu    等于1表示使用gpu，等于0不使用，默认为1

每一轮训练都会保存一下模型，生成checkpoint.txt文件，里面记录上次训练的轮数
每次重新运行脚本都会删除checkpoint.txt和模型文件，所以这只是让你看一下上次训练到哪了

训练完成后会在image文件夹下生成图片，一般来说每轮训练的时候不在val上进行测试
所以图片只会有在训练集上的结果

训练好的模型保存在model文件夹下

训练好模型后使用eval.py进行coco评估，里面有一个阈值需要自己设定
脚本里面默认是阈值是0.5，你需要测试多少阈值就直接设置然后跑就行了

服务器上的硬盘空间不够，所有目前还没有将coco数据集上传


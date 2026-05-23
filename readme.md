# 基于自回归解码与伪标签学习的SVTRv2文本识别算法改进  

## 0.说明  
- 基于[OpenOCR-SVTRv2](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2)  
- 本实验所用硬件环境为8卡 NVIDIA GeForce RTX 3090

## 1.SVTRv2-NRTR模型的训练    
数据集：[原版SVTRv2](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2)中英文混合（训练+测试）数据集
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_nrtr.yml
```
## 2.为无标注数据集打标  
无标注数据集：https://www.modelscope.cn/datasets/yuguohao/STR_Unlabel
```
CUDA_VISIBLE_DEVICES=0 python tools/label_rec_all_en.py --c configs/rec/svtrv2/svtrv2_nrtr.yml
```
生成伪标签数据集：https://www.modelscope.cn/datasets/yuguohao/STR_Labeled
## 3.SVTRv2-CTC的重训练  
训练集：[原版SVTRv2](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2)中英文混合训练数据集(4倍过采样)  +  伪标签数据集  
测试集：[原版SVTRv2](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2)中英文混合测试数据集
```
# stage 1:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_rctc_1.yml

# stage 2:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_1.yml
```


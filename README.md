# yolo-tensorflow
This is the implementation of YOLO v1. Include the complete pretrain on Imagenet, train on VOC 2007/2012 and prediction.
The YOLO network architecture is slightly different with the original paper, which use the 3×3 kernel and 30 filters conv layer to replace the final 2 full connected layer, the reason is I found it's hard to converge using the orginial structure, plus it consume more graphic card memory, if using the conv layer to replace these two full connected layer, it can achieve the similar result with less memory requirement and faster computation speed, and it don't need the dropout and L2 to keep model generality when using data augenmentation.
The whole training process include the below parts:

**1. Pretrain on Imagenet**

**2. Train on VOC2012/2007 dataset**

**3. Prdiction**

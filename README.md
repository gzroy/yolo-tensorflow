# yolo-tensorflow
This is the implementation of YOLO v1. Include the complete pretrain on Imagenet, train on VOC 2007/2012 and prediction.
The YOLO network architecture is slightly different with the original paper, which use the 3×3 kernel and 30 filters conv layer to replace the final 2 full connected layer, the reason is I found it's hard to converge using the orginial structure, plus it consume more graphic card memory, if using the conv layer to replace these two full connected layer, it can achieve the similar result with less memory requirement and faster computation speed, and it don't need the dropout and L2 to keep model generality when using data augenmentation.
The whole training process include the below parts:

**1. Pretrain on Imagenet**
Download the Imagenet training and validiation dataset, decompress and place the images on floder "Imagenet/Imagenet/train_imagenet/" and 'Imagenet/Imagenet/val_imagenet/'.
Run the imagenet_preprocess.py to generate the training records and validiation records in TFRECORD format.
Run the imagenet_train.ipynb to train the image feature extractor on Imagnent dataset. It can achieve around TOP-5 85% accuracy on validation dataset.
**2. Train on VOC2012/2007 dataset**
Download the PASCAL VOC 2012 and 2007 dataset. 
Run the pascal_bbox_preprocess.py to parse the bbox information.
Run the pascal_tfrecord_process_yolov2.py to generate the training records and validiation records in TFRECORD format.
Run the yolo_training.ipynb to start training.
**3. Prdiction**
Run the yolo_predict.ipynb to detect object on the picture.

#-*- encoding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
from multiprocessing import Process, Queue
import sys
import time
import random
import math
 
max_num = 1400  #max record number in one file
train_path = '/home/roy/AI/pascal/VOC2012_train/JPEGImages/'  #the folder stroes the train images
#valid_path = 'VOC2012_test/JPEGImages/'  #the folder stroes the validation images
cores = 10   #number of CPU cores to process

resize_width = 500   #416*1.2
resize_height = 500  #416*1.2

grids = 13
 
#VOC2012_train图片共分为20个类别，构建一个字典，Key是类名，value是0-19
labels_dict = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, 'horse':5, 'sheep':6, 'aeroplane':7, 'bicycle':8,
               'boat':9, 'bus':10, 'car':11, 'motorbike':12, 'train':13, 'bottle':14, 'chair':15, 'diningtable':16,
               'pottedplant':17, 'sofa':18, 'tvmonitor':19}

#读取bbox文件
bbox_list = {}
with open('bbox_train.csv', 'r') as bboxfile:
    records = bboxfile.readlines()
    for record in records:
        fields = record.strip().split(',')
        filename = fields[0][:-4]
        labels = [labels_dict[x] for x in fields[1].split(';')]
        xmin = [float(x) for x in fields[2].split(';')]
        ymin = [float(x) for x in fields[3].split(';')]
        xmax = [float(x) for x in fields[4].split(';')]
        ymax = [float(x) for x in fields[5].split(';')]
        bbox_list[filename] = [labels, xmin, ymin, xmax, ymax] 
files = bbox_list.keys()
        
#构建训练集文件列表，里面的每个元素是路径名+图片文件名
train_images_filenames = os.listdir(train_path)
train_images = []
for image_file in train_images_filenames:
    if image_file[:-4] in files:
        train_images.append(train_path+','+image_file)
random.shuffle(train_images)
'''
#构建验证集文件列表，里面的每个元素是路径名+图片文件名
valid_images_filenames = os.listdir(valid_path)
valid_images = []
for image_file in valid_images_filenames:
    if image_file[:-4] in files:
        valid_images.append(valid_path+','+image_file)
'''
#把图像数据和标签转换为TRRECORD的格式
def make_example(image, height, width, label, bbox, filename):
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'height' : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'channels' : tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
        'colorspace' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[colorspace])),
        'img_format' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        'bbox_xmin' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[0])),
        'bbox_xmax' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[2])),
        'bbox_ymin' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[1])),
        'bbox_ymax' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[3])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename]))
    }))
 
#这个函数用来生成TFRECORD文件，第一个参数是列表，每个元素是图片文件名，第二个参数是写入的目录名
#第三个参数是文件名的起始序号，第四个参数是队列名称，用于和父进程发送消息
def gen_tfrecord(trainrecords, targetfolder, startnum, queue):
    tfrecords_file_num = startnum
    file_num = 0
    total_num = len(trainrecords)
    pid = os.getpid()
    queue.put((pid, file_num))
    writer = tf.python_io.TFRecordWriter(targetfolder+"train_"+str(tfrecords_file_num)+".tfrecord")
    for record in trainrecords:
        file_num += 1
        fields = record.split(',')
        img_raw = cv2.imread(fields[0]+fields[1])
        height, width, _ = img_raw.shape
        img = cv2.resize(img_raw, (resize_width, resize_height))
        height_ratio = resize_height/height
        width_ratio = resize_width/width
        img_jpg = cv2.imencode('.jpg', img)[1].tobytes()
        bbox = bbox_list[fields[1][:-4]]
        bbox[1] = [int(item*width_ratio) for item in bbox[1]]   #xmin
        bbox[3] = [int(item*width_ratio) for item in bbox[3]]   #xmax
        bbox[2] = [int(item*height_ratio) for item in bbox[2]]  #ymin
        bbox[4] = [int(item*height_ratio) for item in bbox[4]]  #ymax
        label = bbox[0]
        ex = make_example(img_jpg, resize_height, resize_width, label, bbox[1:], fields[1].encode())
        writer.write(ex.SerializeToString())
        #每写入100条记录，向父进程发送消息，报告进度
        if file_num%100==0:
            queue.put((pid, file_num))
        if file_num%max_num==0:
            writer.close()
            tfrecords_file_num += 1
            writer = tf.python_io.TFRecordWriter(targetfolder+"train_"+str(tfrecords_file_num)+".tfrecord")
    writer.close()        
    queue.put((pid, file_num))
 
#这个函数用来多进程生成TFRECORD文件，第一个参数是要处理的图片的文件名列表，第二个参数是需要用的CPU核心数
#第三个参数写入的文件目录名
def process_in_queues(fileslist, cores, targetfolder):
    total_files_num = len(fileslist)
    each_process_files_num = int(total_files_num/cores)
    files_for_process_list = []
    for i in range(cores-1):
        files_for_process_list.append(fileslist[i*each_process_files_num:(i+1)*each_process_files_num])
    files_for_process_list.append(fileslist[(cores-1)*each_process_files_num:])
    files_number_list = [len(l) for l in files_for_process_list]
    
    each_process_tffiles_num = math.ceil(each_process_files_num/max_num)
    
    queues_list = []
    processes_list = []
    for i in range(cores):
        queues_list.append(Queue())
        #queue = Queue()
        processes_list.append(Process(target=gen_tfrecord, 
                                      args=(files_for_process_list[i],targetfolder,
                                      each_process_tffiles_num*i+1,queues_list[i],)))
 
    for p in processes_list:
        Process.start(p)
 
    #父进程循环查询队列的消息，并且每0.5秒更新一次
    while(True):
        try:
            total = 0
            progress_str=''
            for i in range(cores):
                msg=queues_list[i].get()
                total += msg[1]
                progress_str+='PID'+str(msg[0])+':'+str(msg[1])+'/'+ str(files_number_list[i])+'|'
            progress_str+='\r'
            print(progress_str, end='')
            if total == total_files_num:
                for p in processes_list:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.5)
        except:
            break
    return total
 
if __name__ == '__main__':
    print('Start processing train data using %i CPU cores:'%cores)
    starttime=time.time()       	  
    total_processed = process_in_queues(train_images, cores, targetfolder='train_tf/')
    endtime=time.time()
    print('\nProcess finish, total process %i images in %i seconds'%(total_processed, int(endtime-starttime)), end='')
    ''' 
    print('Start processing validation data using %i CPU cores:'%cores)
    starttime=time.time()  
    total_processed = process_in_queues(valid_images, cores, targetfolder='test_tf_v1/')
    endtime=time.time()
    print('\nProcess finish, total process %i images, using %i seconds'%(total_processed, int(endtime-starttime)), end='')
    '''
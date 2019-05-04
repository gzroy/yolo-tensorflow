import tensorflow as tf

def _conv(name, inputs, kernel_size, in_channels, out_channels, stride, padding, trainable, bias_init, training):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(shape=[kernel_size,kernel_size,in_channels,out_channels], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False), trainable=trainable, name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1,stride,stride,1], padding=padding)
        biases = tf.get_variable(initializer=tf.constant(bias_init, shape=[out_channels], dtype=tf.float32), trainable=trainable, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        output = tf.nn.leaky_relu(bias, alpha=0.1, name=name)
        output_bn = tf.layers.batch_normalization(output, axis=3, name='bn', trainable=trainable, training=training, reuse=tf.AUTO_REUSE)
        return output_bn

def inference(images, pretrain_trainable=True, wd=0.0005, pretrain_training=True, yolo_training=True):
    conv1 = _conv('conv1', images, 7, 3, 64, 2, 'SAME', pretrain_trainable, 0.01, pretrain_training)       #112*112*64
    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool1')   #56*56*64
    conv2 = _conv('conv2', pool1, 3, 64, 192, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)      #56*56*192
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool2')   #28*28*192
    conv3 = _conv('conv3', pool2, 1, 192, 128, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #28*28*128
    conv4 = _conv('conv4', conv3, 3, 128, 256, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #28*28*256
    conv5 = _conv('conv5', conv4, 1, 256, 256, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #28*28*256
    conv6 = _conv('conv6', conv5, 3, 256, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #28*28*512
    pool6 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool6')   #14*14*512
    conv7 = _conv('conv7', pool6, 1, 512, 256, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #14*14*256
    conv8 = _conv('conv8', conv7, 3, 256, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #14*14*512
    conv9 = _conv('conv9', conv8, 1, 512, 256, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)     #14*14*256
    conv10 = _conv('conv10', conv9, 3, 256, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)   #14*14*512
    conv11 = _conv('conv11', conv10, 1, 512, 256, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)  #14*14*256
    conv12 = _conv('conv12', conv11, 3, 256, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)  #14*14*512
    conv13 = _conv('conv13', conv12, 1, 512, 256, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)  #14*14*256
    conv14 = _conv('conv14', conv13, 3, 256, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)  #14*14*512
    conv15 = _conv('conv15', conv14, 1, 512, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training)  #14*14*512
    conv16 = _conv('conv16', conv15, 3, 512, 1024, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training) #14*14*1024
    pool16 = tf.nn.max_pool(conv16, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool16')  #7*7*1024
    conv17 = _conv('conv17', pool16, 1, 1024, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training) #7*7*512
    conv18 = _conv('conv18', conv17, 3, 512, 1024, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training) #7*7*1024
    conv19 = _conv('conv19', conv18, 1, 1024, 512, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training) #7*7*512
    conv20 = _conv('conv20', conv19, 3, 512, 1024, 1, 'SAME', pretrain_trainable, 0.01, pretrain_training) #7*7*1024

    #For YOLO training, add below 4 conv layers and 2 full connect layers.
    #Remember to set the above pretrained 20 conv layers to non trainable.
    if not pretrain_training:
        new_conv21 = _conv('new_conv21', conv20, 3, 1024, 1024, 1, 'SAME', True, 0.01, yolo_training) #14*14*1024
        new_conv22 = _conv('new_conv22', new_conv21, 3, 1024, 1024, 2, 'SAME', True, 0.01, yolo_training) #7*7*1024
        new_conv23 = _conv('new_conv23', new_conv22, 3, 1024, 1024, 1, 'SAME', True, 0.01, yolo_training) #7*7*1024
        new_conv24 = _conv('new_conv24', new_conv23, 3, 1024, 1024, 1, 'SAME', True, 0.01, yolo_training) #7*7*1024
        new_conv25 = _conv('new_conv25', new_conv24, 3, 1024, 30, 1, 'SAME', True, 0.01, yolo_training) #7*7*30
        return new_conv25
    #For Imagenet pretrain
    else:
        avg_layer = tf.reduce_mean(conv20, axis=[1,2], keepdims=True)    #1024
        flatten = tf.layers.flatten(inputs=avg_layer, name='flatten')
        with tf.variable_scope('local', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(initializer=tf.truncated_normal([1024,1000], dtype=tf.float32, stddev=1/(1000)), trainable=True, name='weights')
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable(initializer=tf.constant(1.0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')
            local = tf.nn.xw_plus_b(flatten, weights, biases, name='local')
        return local
    

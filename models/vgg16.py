import tensorflow as tf
dropout = tf.layers.dropout
conv2d = tf.layers.conv2d
max_pooling2d = tf.layers.max_pooling2d
dense = tf.layers.dense
flatten = tf.layers.flatten
relu = tf.nn.relu


_imagenet_hdim = [64,  64,
                  128, 128,
                  256, 256, 256,
                  512, 512, 512, 
                  512, 512, 512,
                  4096, 4096]


def vgg16(img_shape, num_classes, is_training, 
          weights=None, droprate=0.5, input_tensor=None,
          hdim=_imagenet_hdim):

    if weights is not None:
        return vgg16_pretrained(img_shape, num_classes, is_training,
                                weights, droprate, input_tensor)

    if input_tensor is None:
        x = tf.placeholder(tf.float32, shape=(-1,) + img_shape)
        x = tf.reshape(x, shape=(-1,) + img_shape)
    else:
        x = tf.reshape(input_tensor, shape=(-1,) + img_shape)

    net = conv2d(x,   hdim[0], 3, activation=relu, name='conv11')
    net = conv2d(net, hdim[1], 3, activation=relu, name='conv12')
    net = max_pooling2d(net, 2, 2, name='pool1')

    net = conv2d(net, hdim[2], 3, activation=relu, name='conv21')
    net = conv2d(net, hdim[3], 3, activation=relu, name='conv22')
    net = max_pooling2d(net, 2, 2, name='pool2')

    net = conv2d(net, hdim[4], 3, activation=relu, name='conv31')
    net = conv2d(net, hdim[5], 3, activation=relu, name='conv32')
    net = conv2d(net, hdim[6], 3, activation=relu, name='conv33')
    net = max_pooling2d(net, 2, 2, name='pool3')

    net = conv2d(net, hdim[7], 3, activation=relu, name='conv41')
    net = conv2d(net, hdim[8], 3, activation=relu, name='conv42')
    net = conv2d(net, hdim[9], 3, activation=relu, name='conv43')
    net = max_pooling2d(net, 2, 2, name='pool4')

    net = conv2d(net, hdim[10], 3, activation=relu, name='conv51')
    net = conv2d(net, hdim[11], 3, activation=relu, name='conv52')
    net = conv2d(net, hdim[12], 3, activation=relu, name='conv53')
    net = max_pooling2d(net, 2, 2, name='pool5')

    net = flatten(net)

    net = dense(net, hdim[13], activation=relu, name='fc6')
    net = dropout(net, rate=droprate, training=is_training)

    net = dense(net, hdim[14], activation=relu, name='fc7')
    net = dropout(net, rate=droprate, training=is_training)

    logits = dense(net, num_classes, activation=None, name='fc8')
    return logits, x


pretrained_parameter_names = [
    'conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 
    'conv2_1_W','conv2_1_b','conv2_2_W', 'conv2_2_b', 
    'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 
    'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 
    'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b', 
    'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']


def vgg16_pretrained(img_shape, num_classes, is_training, 
                     weights=None, droprate=0.5, input_tensor=None):

    ijs = [11,12,21,22,31,32,33,41,42,43,51,51,52]
    hdim = (len(weights['conv%s_%s_b' % (ij[0],ij[1])]) for ij in ijs) + 
        (len(weights['fc%s_b' % l]) for l in [6,7,8])

    if input_tensor is None:
        x = tf.placeholder(tf.float32, shape=(-1,) + img_shape)
        net = tf.reshape(x, shape=(-1,) + img_shape)
    else:
        x = tf.reshape(input_tensor, shape=(-1,) + img_shape)
        net = x

    net = conv2d(net,   hdim[0], 3, activation=relu, name='conv11', trainable=False,
     kernel_initializer=weights['conv1_1_W'], bias_initializer=weights['conv1_1_b'])
    net = conv2d(net, hdim[1], 3, activation=relu, name='conv12', trainable=False,
     kernel_initializer=weights['conv1_2_W'], bias_initializer=weights['conv1_2_b'])
    net = max_pooling2d(net, 2, 2, name='pool1')

    net = conv2d(net, hdim[2], 3, activation=relu, name='conv21', trainable=False,
     kernel_initializer=weights['conv2_1_W'], bias_initializer=weights['conv2_1_b'])
    net = conv2d(net, hdim[3], 3, activation=relu, name='conv22', trainable=False,
     kernel_initializer=weights['conv2_2_W'], bias_initializer=weights['conv2_2_b'])
    net = max_pooling2d(net, 2, 2, name='pool2')

    net = conv2d(net, hdim[4], 3, activation=relu, name='conv31', trainable=False,
     kernel_initializer=weights['conv3_1_W'], bias_initializer=weights['conv3_1_b'])
    net = conv2d(net, hdim[5], 3, activation=relu, name='conv32', trainable=False,
     kernel_initializer=weights['conv3_2_W'], bias_initializer=weights['conv3_2_b'])
    net = conv2d(net, hdim[6], 3, activation=relu, name='conv33', trainable=False,
     kernel_initializer=weights['conv3_3_W'], bias_initializer=weights['conv3_3_b'])
    net = max_pooling2d(net, 2, 2, name='pool3')

    net = conv2d(net, hdim[7], 3, activation=relu, name='conv41', trainable=False,
     kernel_initializer=weights['conv4_1_W'], bias_initializer=weights['conv4_1_b'])
    net = conv2d(net, hdim[8], 3, activation=relu, name='conv42', trainable=False,
     kernel_initializer=weights['conv4_2_W'], bias_initializer=weights['conv4_2_b'])
    net = conv2d(net, hdim[9], 3, activation=relu, name='conv43', trainable=False,
     kernel_initializer=weights['conv4_3_W'], bias_initializer=weights['conv4_3_b'])
    net = max_pooling2d(net, 2, 2, name='pool4')

    net = conv2d(net, hdim[10], 3, activation=relu, name='conv51', trainable=False,
     kernel_initializer=weights['conv5_1_W'], bias_initializer=weights['conv5_1_b'])
    net = conv2d(net, hdim[11], 3, activation=relu, name='conv52', trainable=False,
     kernel_initializer=weights['conv5_2_W'], bias_initializer=weights['conv5_2_b'])
    net = conv2d(net, hdim[12], 3, activation=relu, name='conv53', trainable=False,
     kernel_initializer=weights['conv5_3_W'], bias_initializer=weights['conv5_3_b'])
    net = max_pooling2d(net, 2, 2, name='pool5')

    net = flatten(net)

    net = dense(net, hdim[13], activation=relu, name='fc6', trainable=True,
     kernel_initializer=weights['fc6_W'], bias_initializer=weights['fc6_b'])
    net = dropout(net, rate=droprate, training=is_training)

    net = dense(net, hdim[14], activation=relu, name='fc7', trainable=True,
     kernel_initializer=weights['fc7_W'], bias_initializer=weights['fc7_b'])
    net = dropout(net, rate=droprate, training=is_training)

    logits = dense(net, num_classes, activation=None, name='fc8', trainable=True,
     kernel_initializer=weights['fc8_W'], bias_initializer=weights['fc8_b'])
    return logits, x


if __name__ == '__main__':  # test pre-trained model.
    
    import numpy as np
    from imagenet_classes import class_names
    from imageio import imread

    # load test image and weights
    test_image = imread('../test/snake_the_cat.jpg')
    try:
        np.load('vgg16_weights.npz')
    except:
        print("\n\nGet the weights from:\n"
              "http://www.cs.toronto.edu/~frossard/post/vgg16/\n\n")
        raise

    x, logits = vgg16_pretrained(img_shape, 
                       num_classes=, 
                       is_training=False, 
                       weights=)
    y_hat = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        p = sess.run(fetches=[y_hat], feed_dict={x:test_image})
        top10 = sorted(zip(p, range(len(p))))[:10]
        for pi, i in top10:
            print('%s: %s' %(class_names[i], pi))
         
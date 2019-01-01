import tensorflow as tf
dropout = tf.layers.dropout
conv2d = tf.layers.conv2d
max_pooling2d = tf.layers.max_pooling2d
dense = tf.layers.dense
flatten = tf.layers.flatten
relu = tf.nn.relu


def tinycnn(input_tensor, img_shape, num_classes, is_training, droprate=.25):
    # x = tf.placeholder(tf.float32, shape=(-1,) + img_shape)
    x = tf.reshape(input_tensor, shape=(-1,) + img_shape)

    net = conv2d(x, 32, 3, activation=relu, name='conv1')
    net = max_pooling2d(net, 2, 2, name='pool1')

    net = conv2d(net, 64, 3, activation=relu, name='conv2')
    net = max_pooling2d(net, 2, 2, name='pool2')
    net = dropout(net, rate=droprate, training=is_training)

    net = flatten(net)

    net = dense(net, 128, activation=relu, name='fc1')
    net = dropout(net, rate=droprate, training=is_training)

    logits = dense(net, num_classes, name='fc2')
    return logits, x
    
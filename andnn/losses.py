from __future__ import division, print_function, absolute_import
import tensorflow as tf


def ce_wlogits(logits, y):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

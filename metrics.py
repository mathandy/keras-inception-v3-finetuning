from pandas import DataFrame
from sklearn.metrics import confusion_matrix as ugly_confusion_matrix
import tensorflow as tf
from keras.metrics import sparse_top_k_categorical_accuracy


def pretty_confusion_matrix(y_test, y_pred, class_names, abbr=True):
    if abbr:
        class_names = [''.join([w[0] for w in name.split(' ')])
                       for name in class_names]

    cm = DataFrame(data=ugly_confusion_matrix(y_test, y_pred),
                   index=class_names,
                   columns=class_names)
    return cm


def accuracy_fn(y_true, y_pred):
    # https://github.com/keras-team/keras/issues/7818
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.argmax(y_pred, 1)
    correct_predictions = tf.equal(y_pred, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, "float"))


def top_3_error(y_true, y_pred):
    return 1 - sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_error(y_true, y_pred):
    return 1 - sparse_top_k_categorical_accuracy(y_true, y_pred, k=2)

import os
import tensorflow as tf

from core.data import preprocess_image, STRIDE
from core.circle_net import CircleNet, INPUT_SIZE
from core.train import CHECKPOINT_PATH
from core.utils import median, argmaxd

__model__ = CircleNet()


# Loads model into global variable.
def load_model():
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    __model__.load_weights(latest_checkpoint_path)
    return __model__


# Finds circle on the image. Aggregates FCNN's outputs into robust answer.
@tf.function
def find_circle(image, threshold=0.9):
    contains, centers, radiuses = __model__(
        tf.expand_dims(preprocess_image(image), 0))
    contains, centers, radiuses = contains[0], centers[0], radiuses[0]
    height, width = tf.shape(contains)[0], tf.shape(contains)[1]
    meshgrid = tf.stack(tf.meshgrid(tf.range(height, dtype=tf.float32),
                                    tf.range(width, dtype=tf.float32),
                                    indexing='ij'), axis=2)
    original_centers = (
            centers * INPUT_SIZE + meshgrid * STRIDE + (INPUT_SIZE - 1) / 2)
    original_radiuses = radiuses * INPUT_SIZE

    indices = tf.where(tf.squeeze(contains, [2]) > 0.9)
    if tf.equal(tf.size(indices), 0):
        argmax = argmaxd(tf.squeeze(contains, [2]))
        return tf.concat([tf.gather_nd(original_centers, argmax),
                          tf.gather_nd(original_radiuses, argmax)], axis=0)

    mask = tf.squeeze(contains, [2]) > threshold
    n_centers = tf.boolean_mask(original_centers, mask)
    n_radiuses = tf.boolean_mask(original_radiuses, mask)
    return tf.concat([median(n_centers), median(n_radiuses)], axis=0)

import tensorflow as tf
import numpy as np

from core.utils import noisy_circle
from core.circle_net import INPUT_SIZE

STRIDE = 8
IMAGE_SIZE = 200
MAX_RADIUS = 50
NOISE_LEVEL = 2


# Normalizes image.
def preprocess_image(image):
    preprocessed_image = image / (NOISE_LEVEL / 2) - 1
    return tf.cast(tf.expand_dims(preprocessed_image, 2), tf.float32)


# Checks whether point is inside box.
def inside_box(point, box):
    return all(point >= box[:2]) and all(point <= box[2:])


# Calculates l2 distance between points.
def distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


# Checks circle is visible inside window box.
def circle_visible(center, radius, window_box):
    window_center = (window_box[:2] + window_box[2:]) / 2
    return (distance(center, window_center) > radius - INPUT_SIZE / 4
            and (distance(center, window_center) < radius
                 or inside_box(center, window_box)))


# Generates circle data.
def data_generator(positive=True):
    while True:
        params, image = noisy_circle(IMAGE_SIZE, MAX_RADIUS, NOISE_LEVEL)
        center, radius = np.array(params[:2]), np.array(params[2:])
        for _ in range(100):
            bottom = np.random.randint(IMAGE_SIZE - INPUT_SIZE)
            left = np.random.randint(IMAGE_SIZE - INPUT_SIZE)
            bottom_left_corner = np.array([bottom, left])
            window_box = np.array([*bottom_left_corner,
                                   *(bottom_left_corner + INPUT_SIZE - 1)])
            window_center = (window_box[:2] + window_box[2:]) / 2

            if positive ^ (not circle_visible(center, radius, window_box)):
                preprocessed_image = preprocess_image(image)
                relative_radius = radius / INPUT_SIZE
                relative_center = (center - window_center) / INPUT_SIZE
                cls = float(positive)
                yield (
                    {'image_input': preprocessed_image[
                                    bottom:bottom + INPUT_SIZE,
                                    left:left + INPUT_SIZE]},
                    {'contains': tf.constant([[[cls]]]),
                     'center': tf.constant([*relative_center, cls]),
                     'radius': tf.constant([*relative_radius, cls])})


# Creates tensorflow dataset for circle data.
def create_dataset(positive=True):
    return tf.data.Dataset.from_generator(
        data_generator, args=[positive],
        output_types=({'image_input': tf.float32},
                      {'contains': tf.float32,
                       'center': tf.float32,
                       'radius': tf.float32}),
        output_shapes=({'image_input': (INPUT_SIZE, INPUT_SIZE, 1)},
                       {'contains': (1, 1, 1,),
                        'center': (3,),
                        'radius': (2,)}))

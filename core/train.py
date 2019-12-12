import tensorflow as tf

from core.circle_net import CircleNet
from core.data import create_dataset

CHECKPOINT_PATH = './checkpoints/circle-net-{epoch:04d}.ckpt'


# Loss for weighted mean squared error. Last y_true coordinate contains weight.
def weighted_squared_error(y_true, y_pred, eps=1e-7):
    y_pred = tf.squeeze(y_pred, [1, 2])
    coordinates, weight = y_true[:, :-1], y_true[:, -1]
    weighted_sum = tf.reduce_sum(tf.square(coordinates - y_pred),
                                 axis=1) * weight
    return tf.reduce_sum(weighted_sum) / (tf.reduce_sum(weight) + eps)


# Metric for weighted mean absolute error. Last y_true coordinate contains
# weight.
def weighted_absolute_error(y_true, y_pred, eps=1e-7):
    y_pred = tf.squeeze(y_pred, [1, 2])
    coordinates, weight = y_true[:, :-1], y_true[:, -1]
    weighted_sum = tf.reduce_sum(tf.abs(coordinates - y_pred), axis=1) * weight
    return tf.reduce_sum(weighted_sum) / (tf.reduce_sum(weight) + eps)


# Trains CNN to find circle in the image.
def train():
    model = CircleNet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss={'contains': tf.keras.losses.BinaryCrossentropy(),
                        'center': weighted_squared_error,
                        'radius': weighted_squared_error},
                  metrics={'contains': tf.keras.metrics.BinaryAccuracy(),
                           'center': weighted_absolute_error,
                           'radius': weighted_absolute_error},
                  loss_weights={'contains': 1.0, 'center': 10.0,
                                'radius': 10.0})

    positive_dataset = create_dataset(True)
    negative_dataset = create_dataset(False)
    dataset = tf.data.experimental.sample_from_datasets(
        [positive_dataset, negative_dataset],
        [2.0, 1.0])
    dataset = dataset.take(100000)
    dataset = dataset.prefetch(2048).shuffle(1024)
    batched_dataset = dataset.batch(256)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(batched_dataset, epochs=30, callbacks=[cp_callback])


if __name__ == "__main__":
    train()

import tensorflow as tf
import numpy as np
from scipy.signal import savgol_filter
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt


# Draws circle on an image.
def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


# Generates noisy image with circle.
def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


# Calculates median along the zero dimension.
@tf.function
def median(tensor):
    sorted_tensor = tf.sort(tensor, axis=0)
    length = tf.shape(tensor)[0]
    middle = tf.math.floordiv(length, 2)
    if tf.equal(tf.math.floormod(length, 2), 0):
        return (sorted_tensor[middle - 1] + sorted_tensor[middle]) / 2
    else:
        return sorted_tensor[middle]


# Calculates argmax for n-d array.
@tf.function
def argmaxd(tensor):
    return tf.unravel_index(tf.argmax(tf.reshape(tensor, [-1])), tensor.shape)


# Plots loss with learning rates.
def plot_loss(loss, learning_rates):
    plt.figure(figsize=(15, 5))
    plt.title("Loss vs lr")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(learning_rates[0], learning_rates[-1])
    plt.plot(learning_rates, savgol_filter(np.minimum(loss, loss[0]), 41, 3))
    plt.show()


# Helps find perfect learning rate. Paper: https://arxiv.org/abs/1506.01186.
def find_lr(model, dataset, loss):
    start_lr = 1e-5
    end_lr = 0.1
    steps = 100

    initial_weights = model.get_weights()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        start_lr,
        decay_steps=1,
        decay_rate=(end_lr / start_lr) ** (1 / steps),
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(dataset.batch(64).take(1), epochs=steps, verbose=0)
    model.set_weights(initial_weights)

    learning_rates = [lr_schedule(step) for step in range(steps)]
    return history['loss'], learning_rates

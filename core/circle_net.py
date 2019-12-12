from tensorflow.keras.layers import (Conv2D, Input, MaxPool2D, PReLU)
from tensorflow.keras import Model

INPUT_SIZE = 48


# Fully convolutional network, designed to find circles.
def CircleNet():
    image_input = Input(shape=[INPUT_SIZE, INPUT_SIZE, 1], name='image_input')
    layer = Conv2D(48, (5, 5), strides=1, padding='valid')(image_input)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='valid')(layer)
    layer = Conv2D(64, (3, 3), strides=1, padding='valid')(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='valid')(layer)
    layer = Conv2D(96, (5, 5), strides=1, padding='valid')(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='valid')(layer)
    layer = Conv2D(256, (3, 3), strides=1, padding='valid')(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = Conv2D(512, (1, 1), strides=1, padding='valid')(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)

    contains = Conv2D(1, (1, 1), strides=1, activation='sigmoid',
                      name='contains')(layer)
    center = Conv2D(2, (1, 1), strides=1, name='center')(layer)
    radius = Conv2D(1, (1, 1), strides=1, name='radius')(layer)
    model = Model(inputs=[image_input],
                  outputs=[contains, center, radius])

    return model

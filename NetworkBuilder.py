import tensorflow as tf

class NetworkBuilder:
    def __init__(self):
        pass
    def attach_conv_layer(self, input_layer, output_size=32, feature_size=[5, 5], strides=[1, 1, 1, 1], padding='SAME',
                          summary=False):
        input_size = input_layer.get_shape().as_list()[-1]
        conv_weights = tf.Variable(tf.random_normal([feature_size[0], feature_size[1], input_size, output_size]),
                                   name='conv_weights')
        conv_biases = tf.Variable(tf.random_normal([output_size], name='canv_biases'))
        conv_layer = tf.nn.conv2d(input_layer, conv_weights, strides, padding, name='conv_layer')
        if summary:
            tf.summary.histogram(conv_weights.name, conv_weights)
        return conv_layer

    def attach_pooling_layer(self):
        pass
    def attach_flatten_layer(self):
        pass
    def attach_dense_layer(self):
        pass
    def attach_relu_layer(self):
        pass
    def attach_softmax_layer(self):
        pass
    def attach_sigmoid_layer(self):
        pass


from abc import ABC

import numpy as np
import tensorflow as tf


class Conv2DBN(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', use_bias=True, bn=True, activation=True):
        super(Conv2DBN, self).__init__()
        self.bn = bn
        self.activation = activation
        self.conv2d = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             use_bias=use_bias,
                                             kernel_regularizer=tf.keras.regularizers.l2())
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv2d(inputs)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        return x


class ChannelShuffle(tf.keras.layers.Layer):
    '''
    '''

    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.num_groups = num_groups

    def call(self, inputs, **kwargs):
        if len(inputs.shape) != 4:
            raise Exception("inputs dimension must be 4.")
        n, h, w, c = inputs.shape
        inputs_reshaped = tf.reshape(inputs, [-1, h, w, self.num_groups, c // self.num_groups])
        inputs_transposed = tf.transpose(inputs_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(inputs_transposed, [-1, h, w, c])
        return output


class AALayer(tf.keras.layers.Layer):
    """
    Attention Aggregate Layer
    inputs shape: (b,n,k,c)
    outputs shape: (b,n,1,c)
    """

    def __init__(self, filters, groups=3):
        super(AALayer, self).__init__()
        self.filters = filters
        self.groups = groups

    def build(self, input_shape):
        b, n, k, c = input_shape
        # l_agg
        self.l_channel_shuffle = ChannelShuffle(num_groups=self.groups)
        self.l_aggregate = tf.keras.layers.Conv2D(filters=c, kernel_size=(1, k), strides=(1, 1), padding='valid', groups=self.groups)
        self.l_aggregate_bn = tf.keras.layers.BatchNormalization()
        self.l_dense = tf.keras.layers.Dense(units=c, activation=None, use_bias=False)
        # f_agg
        self.f_dense = tf.keras.layers.Dense(units=c, activation=None, use_bias=False)
        #
        self.conv = Conv2DBN(filters=self.filters, kernel_size=1)

    def call(self, inputs, training=None, **kwargs):
        b, n, k, c = inputs.shape

        # local pooling
        l_agg = self.l_aggregate(self.l_channel_shuffle(inputs))  # b*n*1*filters
        l_agg = self.l_aggregate_bn(l_agg,training=training)
        l_agg = tf.nn.leaky_relu(l_agg)
        f_reshaped = tf.reshape(l_agg, shape=[-1, n, c])
        att_activation = self.l_dense(f_reshaped)
        att_scores = tf.nn.softmax(att_activation, axis=1)
        l_agg = f_reshaped * att_scores  # -1*k*f
        l_agg = tf.reshape(l_agg, shape=[-1, n, 1, c])  # b*n*1*filters

        # attention pooling
        f_reshaped = tf.reshape(inputs, shape=[-1, k, c])
        att_activation = self.f_dense(f_reshaped)
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores  # -1*k*f
        f_agg = tf.reshape(f_agg, [-1, n, k, c])  # b*n*k*filters
        # max pooling
        max_fagg = tf.reduce_max(f_agg, axis=2,keepdims=True)  #  b*n*1*filters
        # avg pooling
        mean_fagg = tf.reduce_mean(f_agg, axis=2,keepdims=True)  # b*n*1*filters
        f_agg = tf.reduce_sum(f_agg, axis=-2,keepdims=True)  # b*n*1*filters
        agg = tf.concat([f_agg ,l_agg, max_fagg, mean_fagg], -1)  # b*n*1*4filter
        agg = self.conv(agg, training=training)
        return agg  # b*n*1*c


class PairwiseDistance(tf.keras.layers.Layer):
    def __init__(self, k=20):
        super(PairwiseDistance, self).__init__()
        self.k = int(k)

    def call(self, inputs, **kwargs):
        inputs = tf.squeeze(inputs)  # (B,N,F)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=0)
        point_cloud_transpose = tf.transpose(inputs, perm=[0, 2, 1])  # (B,F,N)
        point_cloud_inner = tf.matmul(inputs, point_cloud_transpose)  # (B,N,N)
        point_cloud_square = tf.square(inputs)
        point_cloud_square = tf.reduce_sum(point_cloud_square, axis=-1, keepdims=True)  # (B,N,1)
        point_cloud_square_transpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])  # (B,1,N)

        return point_cloud_square - 2 * point_cloud_inner + point_cloud_square_transpose  # (B,N,N)


class KNN(tf.keras.layers.Layer):
    def __init__(self, k=20):
        super(KNN, self).__init__()
        self.k = k

    def call(self, inputs, **kwargs):
        neg_adj = -1 * inputs
        _, nn_idx = tf.math.top_k(neg_adj, k=self.k, sorted=True)
        return nn_idx


class EdgeFeature(tf.keras.layers.Layer):
    def __init__(self, k=20):
        super(EdgeFeature, self).__init__()
        self.k = k

    def call(self, inputs, **kwargs):  # inputs is [point_cloud, nn_idx]
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list.')
        if len(inputs) != 2:
            raise ValueError('inputs must be include two elements.')

        point_cloud, nn_idx = inputs

        point_cloud = tf.squeeze(point_cloud)  # (B,N,F)
        if len(point_cloud.shape) == 2:
            point_cloud = tf.expand_dims(point_cloud, axis=0)
        point_cloud_central = point_cloud
        batch_size = point_cloud_central.shape[0]
        num_points = point_cloud_central.shape[1]
        num_attributes = point_cloud_central.shape[2]

        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1])
        point_cloud_flat = tf.reshape(point_cloud, [-1, num_attributes])  # (B*N,F)
        # nn_idx+idx_:(B,N,K)+(B,1,1) -> (B,N,K)
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)  # (B,N,K,F)
        point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)  # (B,N,1,F)
        point_cloud_central = tf.tile(point_cloud_central, [1, 1, self.k, 1])  # (B,N,K,F)
        point_cloud_relative = point_cloud_neighbors - point_cloud_central
        point_cloud_distance = tf.reduce_sum(tf.square(point_cloud_relative), -1, keepdims=True)
        edge_feature = tf.concat(
            [point_cloud_central, point_cloud_neighbors,point_cloud_relative, point_cloud_distance],
            axis=-1)  # (B,N,K,3F+1)

        return edge_feature


class DGCNN(tf.keras.Model, ABC):

    def __init__(self, k=30, num_classes=12):
        super(DGCNN, self).__init__()

        self.adj_0 = PairwiseDistance(k)
        self.nn_dix_0 = KNN(k)
        self.edge_feature_0 = EdgeFeature(k)
        self.out1 = Conv2DBN(60, 1)
        self.out2 = Conv2DBN(60, 1)
        self.aalayer0= AALayer(60)

        self.adj_1 = PairwiseDistance(k)
        self.nn_dix_1 = KNN(k)
        self.edge_feature_1 = EdgeFeature(k)
        self.out3 = Conv2DBN(60, 1)
        self.out4 = Conv2DBN(60, 1)
        self.aalayer1 = AALayer(60)

        self.adj_2 = PairwiseDistance(k)
        self.nn_dix_2 = KNN(k)
        self.edge_feature_2 = EdgeFeature(k)
        self.out5 = Conv2DBN(60, 1)
        # self.out6 = Conv2DBN(60, 1)
        self.aalayer2 = AALayer(60)

        self.out7 = Conv2DBN(1024, 1)

        self.conv1 = Conv2DBN(512, 1)
        self.conv2 = Conv2DBN(256, 1)
        self.dropout = tf.keras.layers.Dropout(rate=0.30)
        self.output_layer = Conv2DBN(num_classes, 1, bn=False, activation=False)

    def call(self, inputs, training=None, mask=None):
        input_image = tf.expand_dims(inputs, -1)  # (B,N,F)
        num_points_one_batch = input_image.shape[1]

        adj = self.adj_0(inputs[:, :, 3:])
        nn_idx = self.nn_dix_0(adj)
        x = self.edge_feature_0([input_image, nn_idx])  # (B,N,K,F)
        x = self.out1(x, training=training)
        x = self.out2(x, training=training)
        net_1 = self.aalayer0(x, training=training)
        # net_1 = tf.reduce_max(x, axis=-2, keepdims=True)  # (B,N,1,64)

        adj = self.adj_1(net_1)
        nn_idx = self.nn_dix_1(adj)
        x = self.edge_feature_1([net_1, nn_idx])  # (B,N,K,F)
        x = self.out3(x, training=training)
        x = self.out4(x, training=training)
        net_2 = self.aalayer1(x, training=training)
        # net_2 = tf.reduce_max(x, axis=-2, keepdims=True)  # (B,N,1,64)

        adj = self.adj_1(net_2)
        nn_idx = self.nn_dix_2(adj)
        x = self.edge_feature_2([net_2, nn_idx])  # (B,N,K,F)
        x = self.out5(x, training=training)
        net_3 = self.aalayer2(x,training=training)
        # net_3 = tf.reduce_max(x, axis=-2, keepdims=True)  # (B,N,1,64)

        x = self.out7(tf.concat([net_1, net_2, net_3], axis=-1), training=training)
        x = tf.nn.max_pool2d(x, [1, x.shape[1], 1, 1], strides=1, padding='VALID')  # (B,1,1,1024)
        x = tf.tile(x, [1, num_points_one_batch, 1, 1])  # (B,N,1,1024)
        x = tf.concat([x, net_1, net_2, net_3], 3)  # (B,N,1,1024+3*64)

        x = self.conv1(x, training=training)  # (B,N,1,512)
        x = self.conv2(x, training=training)  # (B,N,1,256)
        x = self.dropout(x, training=training)
        x = self.output_layer(x, training=training)  # (B,N,1,13)
        x = tf.squeeze(x, [2])  # (B,N,13)

        return x


class DGCNNLoss(tf.keras.losses.Loss, ABC):

    def __init__(self, reduction='AUTO'):
        if reduction == 'NONE':
            super(DGCNNLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        else:
            super(DGCNNLoss, self).__init__()

    def call(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

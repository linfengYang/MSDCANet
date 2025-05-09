from Arguments import *
from Logger import log
from tensorflow.keras.models import Model  # Input,
from tensorflow.keras.layers import Dense, Dropout,SeparableConv2D,GRU,Bidirectional, Conv2D, GlobalMaxPooling2D, Flatten, Reshape, Lambda, dot, \
    UpSampling2D, UpSampling1D,Multiply,ReLU, Add, Concatenate, Activation, concatenate, Conv1D, SpatialDropout1D, BatchNormalization, add,MaxPooling2D,\
    LayerNormalization,TimeDistributed
from tensorflow.keras.utils import plot_model  # print_summary,
import numpy as np
import tensorflow.keras.backend as K
import os
from tensorflow.keras.activations import relu,sigmoid
from typing import List, Tuple
import tensorflow.keras.backend as K
# #import keras.layers
# from tensorflow.keras import optimizers
# from tensorflow.keras.engine.topology import Layer
# import tensorflow as tf # if tensorflow 1
# import tensorflow.compat.v1 as tf # if using tensorflow 2
# tf.disable_v2_behavior()
import tensorflow._api.v2.compat.v1 as tf


tf.disable_v2_behavior()

########################
import h5py
import argparse
from tensorflow.keras.layers import Layer,GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D, Reshape, Dense, multiply, Permute, \
    Concatenate,Conv2D, Conv1D,Add, DepthwiseConv2D,GlobalAveragePooling1D,Activation, Lambda,ZeroPadding1D
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid


# import tensorflow  as tf

# Model setting begin, used in Sequence to point Learning based on bidirectional GRU and PAN network for nilm  
# --------------------------------
# -------------seq2point baseline




def DenseBlock(x, num_layers, growth_rate):
    features = [x]
    kernel_sizes = [3, 5, 7]

    for i in range(num_layers):
        out = BatchNormalization()(x)
        # out = AFN(units=x.shape[-1])(x)
        out = ReLU()(out)
        kernel_size = kernel_sizes[i % len(kernel_sizes)]
        out = Conv1D(filters=growth_rate, kernel_size=kernel_size, padding='same')(out)

        # dcnv4_layer = DCNv4_1D(
        #     in_channels=x.shape[-1],
        #     out_channels=growth_rate,
        #     kernel_size=kernel_size
        # )
        # out = dcnv4_layer(out)

        #out = Conv1D(filters=growth_rate, kernel_size=3, padding='same')(out)
        features.append(out)  # ------
        x = Concatenate()(features)  # ------
    return x

def GatedConvBlock(x,filters=16):
    # residual_input
    residual_input = x
    # 第一路径的卷积,主控分支
    conv1_output = Conv1D(filters=filters, kernel_size=7, strides=1, padding='same', dilation_rate=32)(x)
    # 第二路径的卷积
    conv2_output = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    # 卷积增强部分，结合门控机制 门控分支，不建议
    gate_conv1 = Conv1D(filters=filters, kernel_size=7, strides=1, padding='same', dilation_rate=32)(x)
    gate_conv2 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    # 通过sigmoid生成门控信号
    # gate1 = Activation('sigmoid')(gate_conv1)
    # gate2 = Activation('sigmoid')(gate_conv2)

    gate1 = sigmoid(gate_conv1)
    gate2 = sigmoid(gate_conv2)

    # 通过门控信号调制卷积输出
    gated_conv1_output = Multiply()([conv1_output, gate1])
    gated_conv2_output = Multiply()([conv2_output, gate2])

    # 应用激活函数
    gelu1_output = relu(gated_conv1_output)
    gelu2_output = relu(gated_conv2_output)

    # 交叉乘法操作
    cross1 = gelu1_output * gated_conv2_output
    cross2 = gelu2_output * gated_conv1_output

    # 两个交叉项相加
    added_output = Add()([cross1, cross2])

    # 最后通过一个 Conv1D 层，进一步卷积增强
    final_output = Conv1D(filters=filters, kernel_size=5, padding='same')(added_output)
    # final_output = DSCN1D(filters=filters, kernel_size=5, padding='same')(added_output)

    # final_output = Add()([final_output, residual_input])

    return final_output



def InteractiveConvBlock(x):
    # 残差
    residual_input = x
    # 第一路径的卷积
    conv1_output = Conv1D(filters=16, kernel_size=7, strides=1, padding='same')(x)
    # 第二路径的卷积
    # conv1_output = BatchNormalization()(conv1_output)
    conv2_output = Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(x)

    # 应用 relu 激活函数(卷积gate机制)
    gelu1_output = relu(conv1_output)
    gelu2_output = relu(conv2_output)

    # 交叉乘法操作
    cross1 = gelu1_output * conv2_output
    cross2 = gelu2_output * conv1_output

    # 两个交叉项相加
    added_output = Add()([cross1, cross2])

    # 最后通过一个 Conv1D 层
    final_output = Conv1D(filters=16, kernel_size=5, padding='same')(added_output)  # 可以选择适当的参数
    # final_output= Add()([final_output,residual_input])

    return final_output



def multi_scale_attentionDep(inputs, num_heads=8, kernel_sizes=[3, 7]):
    # Step 1: 使用 1D 卷积生成 Q、K、V
    Q = Conv1D(num_heads, 1, padding="same")(inputs)
    K = Conv1D(num_heads, 1, padding="same")(inputs)
    V = Conv1D(num_heads, 1, padding="same")(inputs)

    # Step 2: 对 K 和 V 进行多尺度卷积
    multi_scale_K = []
    multi_scale_V = []
    for kernel_size in kernel_sizes:
        # 使用 DepthwiseConv1D 对 K 和 V 进行多尺度特征提取
        K_scaled = DepthwiseConv1D(kernel_size, padding="same")(K)
        V_scaled = DepthwiseConv1D(kernel_size, padding="same")(V)

        # 使用 1x1 卷积（逐点卷积）恢复通道维度
        K_scaled = Conv1D(num_heads, 1, padding="same")(K_scaled)
        V_scaled = Conv1D(num_heads, 1, padding="same")(V_scaled)

        multi_scale_K.append(K_scaled)
        multi_scale_V.append(V_scaled)

    # Step 3: 对每个尺度的 Q、K、V 计算 ReLU Global Attention
    attention_outputs = []
    for i in range(len(kernel_sizes)):
        # 计算 ReLU-based 注意力分数
        attention_scores = ReLU()(tf.matmul(Q, multi_scale_K[i], transpose_b=True))
        attention_scores = attention_scores / (tf.reduce_sum(attention_scores, axis=-1, keepdims=True) + 1e-6)

        # 使用注意力分数加权 V
        weighted_V = tf.matmul(attention_scores, multi_scale_V[i])
        attention_outputs.append(weighted_V)

    # Step 4: 将多尺度注意力输出在通道维度上拼接
    concatenated_output = Concatenate(axis=-1)(attention_outputs)

    # Step 5: 使用 1x1 卷积将拼接后的多尺度输出映射回输入通道数
    output = Conv1D(inputs.shape[-1], 1, padding="same")(concatenated_output)

    return output




class AFN(Layer):
    def __init__(self, units, **kwargs):
        super(AFN, self).__init__(**kwargs)
        self.units = units

        # 定义可学习的参数
        self.lambda_mu = self.add_weight(name='lambda_mu', shape=(units,), initializer='zeros', trainable=True)
        self.lambda_sigma = self.add_weight(name='lambda_sigma', shape=(units,), initializer='zeros', trainable=True)
        self.lambda_gamma = self.add_weight(name='lambda_gamma', shape=(units,), initializer='zeros', trainable=True)
        self.lambda_beta = self.add_weight(name='lambda_beta', shape=(units,), initializer='zeros', trainable=True)
        self.gamma_bias = self.add_weight(name='gamma_bias', shape=(units,), initializer='ones', trainable=True)
        self.beta_bias = self.add_weight(name='beta_bias', shape=(units,), initializer='zeros', trainable=True)

    def call(self, inputs, training=None):
        # axis=0  == batch normalization   =-1== layer normalization
        mu = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        sigma = tf.math.reduce_std(inputs, axis=-1, keepdims=True)

        # 对输入特征进行归一化
        x_bar = (inputs - mu) / (sigma + 1e-8)

        # 计算缩放和偏移参数
        gamma = self.lambda_gamma * tf.nn.sigmoid(sigma) + self.gamma_bias
        beta = self.lambda_beta * tf.nn.tanh(mu) + self.beta_bias

        # 返回归一化后的特征
        return x_bar * gamma + beta

    def get_config(self):
        config = super(AFN, self).get_config()
        config.update({
            'units': self.units
        })
        return config


# 主模型

# 主模型


def MultiScaleConv1D(input_tensor, filters, kernel_sizes=[3, 5, 7], strides=1):
    convs = []
    for size in kernel_sizes:
        conv = Conv1D(filters=filters, kernel_size=size, padding='same', activation='relu', strides=strides)(input_tensor)
        convs.append(conv)
    return Concatenate()(convs)

# def get_MSDCAmodel(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
#                   cnn='fridge', n_dense=1, pretrainedmodel_dir='./models/'):
#     # 如果输入形状不是 (batch, window_length, 1)，则进行重塑
#     reshape = Reshape((window_length, 1))(input_tensor)
#
#     channel = 8
#     layer1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same', activation='relu')(reshape)
#     # layer1 = SE_Block(layer1)
#     layer2 = Conv1D(filters=channel * 2, kernel_size=3, strides=2, padding='same', activation='relu')(layer1)
#     #layer2 = SE_Block(layer2)
#     layer3 = Conv1D(filters=channel * 4, kernel_size=3, strides=2, padding='same', activation='relu')(layer2)
#     #layer3 = SE_Block(layer3)
#     #layer1 = ResidualMultiScaleConv1D(interactive_1d, filters=channel)
#     # layer2 = ResidualMultiScaleConv1D(layer1, filters=channel * 2)
#     # layer3 = ResidualMultiScaleConv1D(layer2, filters=channel * 4)
#
#     # 横向连接与上采样（类似FPN），使用1D操作
#     layer3_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer3)
#     UP1 = UpSampling1D(size=2)(layer3_lateral)
#     layer2_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer2)
#     layer2_fused = Add()([UP1, layer2_lateral])
#
#     UP2 = UpSampling1D(size=2)(layer2_fused)
#     layer1_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer1)
#     layer1_fused = Add()([UP2, layer1_lateral])
#
#     UP3 = UpSampling1D(size=2)(layer1_fused)
#     layer0_lateral = Conv1D(channel, kernel_size=1, activation='relu')(reshape)
#     interactive_pyramid = Add()([UP3, layer0_lateral])
#
#     # 可选：在融合后进行归一化和激活
#     interactive_pyramid = AFN(units=interactive_pyramid.shape[-1])(interactive_pyramid)
#     interactive_pyramid = ReLU()(interactive_pyramid)
#
#     # 原有的 GatedConvBlock 和 AFN，保持不变
#     interactive_out = GatedConvBlock(interactive_pyramid, 8)  # 改为8通道以适配后续结构
#     interactive_out = AFN(units=interactive_out.shape[-1])(interactive_out)
#     interactive_out = ReLU()(interactive_out)
#
#     # dense_before_pyramid = TimeDistributed(Dense(8, activation='relu'))(interactive_out)
#     # dense_before_pyramid = TimeDistributed(Dense(16, activation='relu'))(dense_before_pyramid)
#
#     # # 将全连接层的输出作为多尺度金字塔结构的输入
#     # # 如果需要保持与后续结构的通道数一致，可以调整 Dense 层的输出维度
#     # interactive_1d = dense_before_pyramid  # 重命名以增强清晰度
#
#     # # ========== 修改后的多尺度金字塔结构 ==========
#
#     # # 不需要重塑为2D，保持为 (batch, window_length, channels)
#     # channel = 8  # 与 GatedConvBlock 输出一致
#     # # interactive_1d = interactive_out  # 重命名以增强清晰度
#
#     # # 类似 Aug 模型的多尺度下采样，使用1D操作
#     # layer1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same', activation='relu')(interactive_1d)
#     # # layer1 = SE_Block(layer1)
#     # layer2 = Conv1D(filters=channel * 2, kernel_size=3, strides=2, padding='same', activation='relu')(layer1)
#     # #layer2 = SE_Block(layer2)
#     # layer3 = Conv1D(filters=channel * 4, kernel_size=3, strides=2, padding='same', activation='relu')(layer2)
#     # #layer3 = SE_Block(layer3)
#     # #layer1 = ResidualMultiScaleConv1D(interactive_1d, filters=channel)
#     # # layer2 = ResidualMultiScaleConv1D(layer1, filters=channel * 2)
#     # # layer3 = ResidualMultiScaleConv1D(layer2, filters=channel * 4)
#
#     # # 横向连接与上采样（类似FPN），使用1D操作
#     # layer3_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer3)
#     # UP1 = UpSampling1D(size=2)(layer3_lateral)
#     # layer2_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer2)
#     # layer2_fused = Add()([UP1, layer2_lateral])
#
#     # UP2 = UpSampling1D(size=2)(layer2_fused)
#     # layer1_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer1)
#     # layer1_fused = Add()([UP2, layer1_lateral])
#
#     # UP3 = UpSampling1D(size=2)(layer1_fused)
#     # layer0_lateral = Conv1D(channel, kernel_size=1, activation='relu')(interactive_1d)
#     # interactive_pyramid = Add()([UP3, layer0_lateral])
#
#
#
#
#     # # 可选：在融合后进行归一化和激活
#     # interactive_pyramid = AFN(units=interactive_pyramid.shape[-1])(interactive_pyramid)
#     # interactive_pyramid = ReLU()(interactive_pyramid)
#
#     # # 无需再次重塑，保持为 (batch, window_length, channels)
#     # interactive_out = interactive_pyramid  # 已经是一维
#
#     # ========== 原有的其他分支与逻辑保持不变 ==========
#     # 多尺度注意力
#     attention_out = multi_scale_attentionDep(reshape)
#
#     # DSCN1d1 = DSCN1D(filters=8, kernel_size=3, strides=1, padding='same')(attention_out)
#     # DSCN1d1 = AFN(units=DSCN1d1.shape[-1])(DSCN1d1)
#     # DSCN1d1 = ReLU()(DSCN1d1)
#
#     # DSCN1d2 = DSCN1D(filters=8, kernel_size=5, strides=1, padding='same')(attention_out)
#     # DSCN1d2 = AFN(units=DSCN1d2.shape[-1])(DSCN1d2)
#     # DSCN1d2 = ReLU()(DSCN1d2)
#
#     # DSCN1d3 = DSCN1D(filters=8, kernel_size=7, strides=1, padding='same')(attention_out)
#     # DSCN1d3 = AFN(units=DSCN1d3.shape[-1])(DSCN1d3)
#     # DSCN1d3 = ReLU()(DSCN1d3)
#
#     # DSCN1d = Concatenate()([DSCN1d1, DSCN1d2, DSCN1d3])
#
#     layerDSCN1d1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same')(attention_out)
#     layerDSCN1d1 = AFN(units=layerDSCN1d1.shape[-1])(layerDSCN1d1)
#     layerDSCN1d1 = ReLU()(layerDSCN1d1)
#
#     layerDSCN1d2 = Conv1D(filters=channel * 2, kernel_size=5, strides=2, padding='same')(layerDSCN1d1)
#     layerDSCN1d2 = AFN(units=layerDSCN1d2.shape[-1])(layerDSCN1d2)
#     layerDSCN1d2 = ReLU()(layerDSCN1d2)
#
#     layerDSCN1d3 = Conv1D(filters=channel * 4, kernel_size=7, strides=2, padding='same')(layerDSCN1d2)
#     layerDSCN1d3 = AFN(units=layerDSCN1d3.shape[-1])(layerDSCN1d3)
#     layerDSCN1d3 = ReLU()(layerDSCN1d3)
#
#     # DSCN1d = Concatenate()([DSCN1d1, DSCN1d2, DSCN1d3])
#
#     layer3_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d3)
#     UP1 = UpSampling1D(size=2)(layer3_lateralDSCN)
#     layer2_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d2)
#     layer2_fusedDSCN = Add()([UP1, layer2_lateralDSCN])
#
#     UP2 = UpSampling1D(size=2)(layer2_fusedDSCN)
#     layer1_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d1)
#     layer1_fusedDSCN = Add()([UP2, layer1_lateralDSCN])
#
#     UP3 = UpSampling1D(size=2)(layer1_fusedDSCN)
#     layer0_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(attention_out)
#     DSCN_pyramid = Add()([UP3, layer0_lateralDSCN])
#
#
#
#     dense_block_out = DenseBlock(DSCN_pyramid, num_layers=3, growth_rate=8)
#     dense_block_out = AFN(units=dense_block_out.shape[-1])(dense_block_out)
#     dense_block_out = ReLU()(dense_block_out)
#
#     # dense_block_out = Concatenate()([dense_block_out, DSCN1d])
#
#     # 融合改进后的 interactive_out 与 dense_block_out
#     fuse_out = Concatenate()([interactive_out, dense_block_out])
#
#     flat = Flatten(name='flatten')(fuse_out)
#     # flat = GlobalAveragePooling1D()(flat)
#     # d1 = Dense(128, activation='relu', name='dense1')(flat)
#     # d1 = Dense(64, activation='relu', name='dense64')(d1)
#     # d1 = Dense(32, activation='relu', name='dense32')(d1)
#
#
#     # d = Dense(window_length, activation='relu', name='dense2')(d1)
#     # d = Add()([d, input_tensor])  # 残差连接回传原始输入
#
#     # d = AFN(units=d.shape[-1])(d)
#
#     #----------
#     d = Dense(128, activation='relu', name='dense2')(flat)
#     if input_tensor.shape[-1] != d.shape[-1]:
#         residual = Dense(units=d.shape[-1], activation=None)(input_tensor)
#     else:
#         residual = input_tensor
#     d = Add()([d, residual])
#     #----------
#     # d = Dense(window_length, activation='relu', name='dense2')(flat)
#     # d = Add()([d, input_tensor])
#
#
#     d = AFN(units=d.shape[-1])(d)
#     d = Dense(64, activation='relu', name='dense1')(d)
#     # d = Dense(64, activation='relu', name='dense64')(d)
#     # d = Dense(32, activation='relu', name='dense32')(d)
#     # d = Dense(16, activation='relu', name='dense16')(d)
#
#
#     d_out = Dense(1, activation='linear', name='output')(d)
#
#     model = Model(inputs=input_tensor, outputs=d_out)
#     return model



def get_MSDCAmodel1(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
                  cnn='fridge', n_dense=1, pretrainedmodel_dir='./models/'):
    # 如果输入形状不是 (batch, window_length, 1)，则进行重塑
    reshape = Reshape((window_length, 1))(input_tensor)

    channel = 8
    layer1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same', activation='relu')(reshape)
    # layer1 = SE_Block(layer1)
    layer2 = Conv1D(filters=channel * 2, kernel_size=5, strides=2, padding='same', activation='relu')(layer1)
    #layer2 = SE_Block(layer2)
    layer3 = Conv1D(filters=channel * 4, kernel_size=7, strides=2, padding='same', activation='relu')(layer2)
    #layer3 = SE_Block(layer3)
    #layer1 = ResidualMultiScaleConv1D(interactive_1d, filters=channel)
    # layer2 = ResidualMultiScaleConv1D(layer1, filters=channel * 2)
    # layer3 = ResidualMultiScaleConv1D(layer2, filters=channel * 4)


    layer3_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer3)
    UP1 = UpSampling1D(size=2)(layer3_lateral)
    layer2_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer2)
    layer2_fused = Add()([UP1, layer2_lateral])

    UP2 = UpSampling1D(size=2)(layer2_fused)
    layer1_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer1)
    layer1_fused = Add()([UP2, layer1_lateral])

    UP3 = UpSampling1D(size=2)(layer1_fused)
    layer0_lateral = Conv1D(channel, kernel_size=1, activation='relu')(reshape)
    interactive_pyramid = Add()([UP3, layer0_lateral])

    # 可选：在融合后进行归一化和激活
    interactive_pyramid = AFN(units=interactive_pyramid.shape[-1])(interactive_pyramid)
    interactive_pyramid = ReLU()(interactive_pyramid)

    # 原有的 GatedConvBlock 和 AFN，保持不变
    interactive_out = GatedConvBlock(interactive_pyramid, 16)  # 改为8通道以适配后续结构
    interactive_out = AFN(units=interactive_out.shape[-1])(interactive_out)
    interactive_out = ReLU()(interactive_out)

    # ========== 原有的其他分支与逻辑保持不变 ==========
    # 多尺度注意力
    attention_out = multi_scale_attentionDep(reshape)


    layerDSCN1d1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same')(attention_out)
    layerDSCN1d1 = AFN(units=layerDSCN1d1.shape[-1])(layerDSCN1d1)
    layerDSCN1d1 = ReLU()(layerDSCN1d1)

    layerDSCN1d2 = Conv1D(filters=channel * 2, kernel_size=5, strides=2, padding='same')(layerDSCN1d1)
    layerDSCN1d2 = AFN(units=layerDSCN1d2.shape[-1])(layerDSCN1d2)
    layerDSCN1d2 = ReLU()(layerDSCN1d2)

    layerDSCN1d3 = Conv1D(filters=channel * 4, kernel_size=7, strides=2, padding='same')(layerDSCN1d2)
    layerDSCN1d3 = AFN(units=layerDSCN1d3.shape[-1])(layerDSCN1d3)
    layerDSCN1d3 = ReLU()(layerDSCN1d3)

    # DSCN1d = Concatenate()([DSCN1d1, DSCN1d2, DSCN1d3])

    layer3_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d3)
    UP1 = UpSampling1D(size=2)(layer3_lateralDSCN)
    layer2_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d2)
    layer2_fusedDSCN = Add()([UP1, layer2_lateralDSCN])

    UP2 = UpSampling1D(size=2)(layer2_fusedDSCN)
    layer1_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d1)
    layer1_fusedDSCN = Add()([UP2, layer1_lateralDSCN])

    UP3 = UpSampling1D(size=2)(layer1_fusedDSCN)
    layer0_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(attention_out)
    DSCN_pyramid = Add()([UP3, layer0_lateralDSCN])



    dense_block_out = DenseBlock(DSCN_pyramid, num_layers=3, growth_rate=16)
    dense_block_out = AFN(units=dense_block_out.shape[-1])(dense_block_out)
    dense_block_out = ReLU()(dense_block_out)

    # dense_block_out = Concatenate()([dense_block_out, DSCN1d])

    # 融合改进后的 interactive_out 与 dense_block_out
    fuse_out = Concatenate()([interactive_out, dense_block_out])

    flat = Flatten(name='flatten')(fuse_out)
    # flat = GlobalAveragePooling1D()(flat)
    # d1 = Dense(128, activation='relu', name='dense1')(flat)
    # d1 = Dense(64, activation='relu', name='dense64')(d1)
    # d1 = Dense(32, activation='relu', name='dense32')(d1)


    # d = Dense(window_length, activation='relu', name='dense2')(d1)
    # d = Add()([d, input_tensor])  # 残差连接回传原始输入

    # d = AFN(units=d.shape[-1])(d)

    #----------
    d = Dense(128, activation='relu', name='dense2')(flat)
    if input_tensor.shape[-1] != d.shape[-1]:
        residual = Dense(units=d.shape[-1], activation=None)(input_tensor)
    else:
        residual = input_tensor
    d = Add()([d, residual])
    #----------
    # d = Dense(window_length, activation='relu', name='dense2')(flat)
    # d = Add()([d, input_tensor])


    d = AFN(units=d.shape[-1])(d)
    # d = Dense(512, activation='relu', name='dense1')(d)
    # d = Dense(256, activation='relu', name='dense64')(d)
    d = Dense(64, activation='relu', name='dense32')(d)
    d = Dense(32, activation='relu', name='dense16')(d)
  

    d_out = Dense(1, activation='linear', name='output')(d)

    model = Model(inputs=input_tensor, outputs=d_out)
    model.summary()

    # print("\nChecking trainable status of each layer:")
    # for layer in model.layers:
    #     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
    # # 添加代码查看不可训练参数的详细列表
    # print("\nNon-trainable parameters:")
    # non_trainable_params = [var for var in model.weights if not var.trainable]
    # for param in non_trainable_params:
    #     print(f"Variable: {param.name}, Shape: {param.shape}")

    return model


def get_MSDCAmodelab1(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
                      cnn='fridge', n_dense=1, pretrainedmodel_dir='./models/'):
    # 如果输入形状不是 (batch, window_length, 1)，则进行重塑
    reshape = Reshape((window_length, 1))(input_tensor)
    channel = 8

    layer1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same', activation='relu')(reshape)
    layer2 = Conv1D(filters=channel * 2, kernel_size=5, strides=2, padding='same', activation='relu')(layer1)
    layer3 = Conv1D(filters=channel * 4, kernel_size=7, strides=2, padding='same', activation='relu')(layer2)

    layer3_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer3)
    UP1 = UpSampling1D(size=2)(layer3_lateral)
    layer2_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer2)
    layer2_fused = Add()([UP1, layer2_lateral])

    UP2 = UpSampling1D(size=2)(layer2_fused)
    layer1_lateral = Conv1D(channel, kernel_size=1, activation='relu')(layer1)
    layer1_fused = Add()([UP2, layer1_lateral])

    UP3 = UpSampling1D(size=2)(layer1_fused)
    layer0_lateral = Conv1D(channel, kernel_size=1, activation='relu')(reshape)
    interactive_pyramid = Add()([UP3, layer0_lateral])

    interactive_pyramid = AFN(units=interactive_pyramid.shape[-1])(interactive_pyramid)
    interactive_pyramid = ReLU()(interactive_pyramid)

    interactive_out = GatedConvBlock(interactive_pyramid, 16)  # 改为8通道以适配后续结构
    interactive_out = AFN(units=interactive_out.shape[-1])(interactive_out)
    interactive_out = ReLU()(interactive_out)

    # ========== 原有的其他分支与逻辑保持不变 ==========
    # 多尺度注意力
    # attention_out = multi_scale_attentionDep(reshape)
    #
    # layerDSCN1d1 = Conv1D(filters=channel, kernel_size=3, strides=2, padding='same')(attention_out)
    # layerDSCN1d1 = AFN(units=layerDSCN1d1.shape[-1])(layerDSCN1d1)
    # layerDSCN1d1 = ReLU()(layerDSCN1d1)
    #
    # layerDSCN1d2 = Conv1D(filters=channel * 2, kernel_size=5, strides=2, padding='same')(layerDSCN1d1)
    # layerDSCN1d2 = AFN(units=layerDSCN1d2.shape[-1])(layerDSCN1d2)
    # layerDSCN1d2 = ReLU()(layerDSCN1d2)
    #
    # layerDSCN1d3 = Conv1D(filters=channel * 4, kernel_size=7, strides=2, padding='same')(layerDSCN1d2)
    # layerDSCN1d3 = AFN(units=layerDSCN1d3.shape[-1])(layerDSCN1d3)
    # layerDSCN1d3 = ReLU()(layerDSCN1d3)
    #
    # # DSCN1d = Concatenate()([DSCN1d1, DSCN1d2, DSCN1d3])
    #
    # layer3_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d3)
    # UP1 = UpSampling1D(size=2)(layer3_lateralDSCN)
    # layer2_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d2)
    # layer2_fusedDSCN = Add()([UP1, layer2_lateralDSCN])
    #
    # UP2 = UpSampling1D(size=2)(layer2_fusedDSCN)
    # layer1_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(layerDSCN1d1)
    # layer1_fusedDSCN = Add()([UP2, layer1_lateralDSCN])
    #
    # UP3 = UpSampling1D(size=2)(layer1_fusedDSCN)
    # layer0_lateralDSCN = Conv1D(channel, kernel_size=1, activation='relu')(attention_out)
    # DSCN_pyramid = Add()([UP3, layer0_lateralDSCN])
    #
    # dense_block_out = DenseBlock(DSCN_pyramid, num_layers=3, growth_rate=16)
    # dense_block_out = AFN(units=dense_block_out.shape[-1])(dense_block_out)
    # dense_block_out = ReLU()(dense_block_out)
    #
    # # fuse_out = Concatenate()([interactive_out, dense_block_out])
    fuse_out = Concatenate()([interactive_out])

    flat = Flatten(name='flatten')(fuse_out)

    d = Dense(128, activation='relu', name='dense2')(flat)
    if input_tensor.shape[-1] != d.shape[-1]:
        residual = Dense(units=d.shape[-1], activation=None)(input_tensor)
    else:
        residual = input_tensor
    d = Add()([d, residual])

    d = AFN(units=d.shape[-1])(d)
    d = Dense(64, activation='relu', name='dense64')(d)
    d = Dense(32, activation='relu', name='dense32')(d)

    d_out = Dense(1, activation='linear', name='output')(d)

    model = Model(inputs=input_tensor, outputs=d_out)
    return model





def squash(vector):
    s_squared_norm = tf.reduce_sum(tf.square(vector), axis=-1, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vector


def gated_depthwise_conv_unit(x, filters, kernel_size, dilation_rate=1, padding='same'):
    """Gated Depthwise Separable Convolution Unit to reduce model size."""
    depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)(x)
    pointwise_conv = Conv2D(filters, kernel_size=(1, 1), padding=padding)(depthwise_conv)

    # Adjust gate to have the same number of filters
    gate = DepthwiseConv2D(kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)(x)
    gate = Conv2D(filters, kernel_size=(1, 1), padding=padding)(gate)

    gated_output = Multiply()([pointwise_conv, tf.keras.activations.sigmoid(gate)])
    return gated_output


def lightweight_rnn_free_recurrent_module(x, filters, kernel_size, dilation_rates):
    """Lightweight RNN-Free Recurrent Module using fewer dilated convolutions."""
    for rate in dilation_rates:
        x_res = gated_depthwise_conv_unit(x, filters=filters, kernel_size=(kernel_size, 1), dilation_rate=rate)

        # Add a Conv2D layer to match the shape if necessary
        if x.shape[-1] != x_res.shape[-1]:
            x_res = Conv2D(x.shape[-1], kernel_size=(1, 1), padding='same')(x_res)

        x = Add()([x, x_res])  # Skip connection for residual learning
    return x




def C2(x, channel):
    layer2 = Conv2D(filters=channel * 2,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(x)
    return layer2


def C21(x, channel, n):
    layer1 = C2(x, channel)
    return layer1


def C_lateral(x, channel):
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    # padding='same',
                    activation='relu',
                    )(x)
    return layer1


def C_up(x, channel):
    layer1 = UpSampling2D((1, 2))(x)
    layer2 = Conv2D(filters=channel,
                    kernel_size=(1, 2),
                    strides=(1, 1),
                    activation='relu',
                    )(layer1)
    return layer2

# -----------------------------------------------
def AugLPN_NILM(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
               cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    channel = 32
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(reshape)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = Add()([UP1, layer3_lateral])
    layer2_lateral = C_lateral(layer2, channel)
    UP2 = UpSampling2D((1, 2))(layer3_)
    layer2_ = Add()([UP2, layer2_lateral])
    layer1_lateral = C_lateral(layer1, channel)
    UP3 = UpSampling2D((1, 2))(layer2_)
    layer1_ = Add()([UP3, layer1_lateral])
    layer1_temp_condv = Conv2D(filters=channel,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               )(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = GlobalMaxPooling2D()(layer1_)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Conv2D(filters=channel,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='relu'
                      )(max_pool)

    max_pool = Activation('hard_sigmoid')(max_pool)
    layer1_mul = multiply([max_pool, layer1_temp_condv])

    layer1_b3 = Conv2D(filters=channel,
                       kernel_size=(1, 3),
                       strides=(1, 1),
                       padding='same'
                       )(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)

    layer1_mul = multiply([layer1_mul, result_1])

    layer2_up = Conv2D(filters=channel * 2,
                       kernel_size=(1, 2),
                       strides=(1, 2)
                       )(layer1_mul)
    layer2_pan_ = Conv2D(filters=channel * 2,
                         kernel_size=(1, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(layer2_)
    layer3_pan = Add()([layer2_up, layer2_pan_])

    layer3_pan_ = Conv2D(filters=channel,
                         kernel_size=(1, 3),
                         strides=(1, 2),
                         padding='same',
                         activation='relu',
                         )(layer3_pan)
    pan_Preoutput = Add()([layer3_pan_, layer3_])

    layer3_pan = Conv2D(filters=96, 
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=2
                        )(layer3_pan)
    layer3_pan = Conv2D(filters=64,  
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=4
                        )(layer3_pan)
    layer3_pan = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan) 
    layer3_pan = Conv2D(filters=32,
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu'
                        )(layer3_pan)

    pan_Preoutput = Add()([layer3_pan, pan_Preoutput]) 
    
    layer3_total_SepConv1 = SeparableConv2D(filters=128, kernel_size=(1, 2),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)
    layer3_total_SepConv2 = SeparableConv2D(filters=128, kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)

    Sep_Add = Add()([layer3_total_SepConv1, layer3_total_SepConv2]) 

    # Dil_Add = Reshape((75, 128), )(Sep_Add)
    # BiGru2 = Bidirectional(GRU(64, return_sequences=True))(Dil_Add)
    # Gru2_output = Reshape((-1, 75, 128), )(BiGru2)

    Gru2_output = Conv2D(filters=64,
                            kernel_size=(1, 2),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            )(Sep_Add)  # Gru2_output
    layer3_total_SepConv1_1 = SeparableConv2D(filters=64, kernel_size=(1, 2),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    layer3_total_SepConv2_2 = SeparableConv2D(filters=64, kernel_size=(1, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    Sep_Add_1 = Add()([layer3_total_SepConv1_1, layer3_total_SepConv2_2]) 
    Sep_Add_1 = Conv2D(filters=64,
                            kernel_size=(1, 2),
                            strides=(1, 2),
                            padding='same',
                            activation='relu',
                            )(Sep_Add_1)
    # Dil_Add_1 = Reshape((75, 64), )(Sep_Add_1)
    # BiGru2_2 = Bidirectional(GRU(32, return_sequences=True))(Dil_Add_1)
    # Gru2_output_1 = Reshape((-1, 75, 64), )(BiGru2_2)

    right_output = Add()([Gru2_output, Sep_Add_1])  # Gru2_output_1
    out1 = Concatenate()([right_output, pan_Preoutput]) 
    flat = Flatten(name='flatten')(out1)
    dense1 = Dense(56, activation='relu', name='dense1')(flat)
    dense1 = Dense(10, activation='relu', name='dense2')(dense1)
    d_out = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = tf.keras.backend.get_session()  # For Tensorflow 2

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    model_def.summary()

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def
# ------------------------------------------------
# -----------------------------------------------
def AugLPN_NILM_16(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    channel = 16
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(reshape)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = Add()([UP1, layer3_lateral])
    layer2_lateral = C_lateral(layer2, channel)
    UP2 = UpSampling2D((1, 2))(layer3_)
    layer2_ = Add()([UP2, layer2_lateral])
    layer1_lateral = C_lateral(layer1, channel)
    UP3 = UpSampling2D((1, 2))(layer2_)
    layer1_ = Add()([UP3, layer1_lateral])

    # ----------PAN----------------------------------------------
    layer1_temp_condv = Conv2D(filters=channel,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               )(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = GlobalMaxPooling2D()(layer1_)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Conv2D(filters=channel,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='relu'
                      )(max_pool)

    max_pool = Activation('hard_sigmoid')(max_pool)
    layer1_mul = multiply([max_pool, layer1_temp_condv])

    layer1_b3 = Conv2D(filters=channel,
                       kernel_size=(1, 3),
                       strides=(1, 1),
                       padding='same'
                       )(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)

    layer1_mul = multiply([layer1_mul, result_1])

    layer2_up = Conv2D(filters=channel * 2,
                       kernel_size=(1, 2),
                       strides=(1, 2)
                       )(layer1_mul)

    layer2_pan_ = Conv2D(filters=channel * 2,
                         kernel_size=(1, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(layer2_)
    layer3_pan = Add()([layer2_up, layer2_pan_])

    layer3_pan_ = Conv2D(filters=channel,
                         kernel_size=(1, 3),
                         strides=(1, 2),
                         padding='same',
                         activation='relu',
                         )(layer3_pan)
    pan_Preoutput = Add()([layer3_pan_, layer3_])

    layer3_pan = Conv2D(filters=48,  
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=2
                        )(layer3_pan)

    layer3_pan = Conv2D(filters=32, 
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=4
                        )(layer3_pan)

    layer3_pan = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan)  
    layer3_pan = Conv2D(filters=channel,
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu'
                        )(layer3_pan)

    pan_Preoutput = Add()([layer3_pan, pan_Preoutput]) 

    layer3_total_SepConv1 = SeparableConv2D(filters=64, kernel_size=(1, 2),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)
    layer3_total_SepConv2 = SeparableConv2D(filters=64, kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)

    Sep_Add = Add()([layer3_total_SepConv1, layer3_total_SepConv2]) 
    Dil_Add = Reshape((75, 64), )(Sep_Add)
    BiGru2 = Bidirectional(GRU(32, return_sequences=True))(Dil_Add)

    Gru2_output = Reshape((-1, 75, 64), )(BiGru2)
    Gru2_output = Conv2D(filters=32,
                         kernel_size=(1, 2),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(Gru2_output)

    layer3_total_SepConv1_1 = SeparableConv2D(filters=32, kernel_size=(1, 2),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    layer3_total_SepConv2_2 = SeparableConv2D(filters=32, kernel_size=(1, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)

    Sep_Add_1 = Add()([layer3_total_SepConv1_1, layer3_total_SepConv2_2]) 
    Sep_Add_1 = Conv2D(filters=32,
                       kernel_size=(1, 2),
                       strides=(1, 2),
                       padding='same',
                       activation='relu',
                       )(Sep_Add_1)

    Dil_Add_1 = Reshape((75, 32), )(Sep_Add_1)
    BiGru2_2 = Bidirectional(GRU(16, return_sequences=True))(Dil_Add_1)

    Gru2_output_1 = Reshape((-1, 75, 32), )(BiGru2_2)


    right_output = Add()([Gru2_output, Gru2_output_1])

    out1 = Concatenate()([right_output, pan_Preoutput])  

    flat = Flatten(name='flatten')(out1)
    dense1 = Dense(28, activation='relu', name='dense1')(flat)
    # dense1 = Dropout(0.1)(dense1)
    dense1 = Dense(10, activation='relu', name='dense2')(dense1)
    d_out = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = tf.keras.backend.get_session()  # For Tensorflow 2

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    model_def.summary()

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def
# ------------------------------------------------

# -----------------------------------------------
def AugLPN_NILM_48(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    channel = 48
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(reshape)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = Add()([UP1, layer3_lateral])
    layer2_lateral = C_lateral(layer2, channel)
    UP2 = UpSampling2D((1, 2))(layer3_)
    layer2_ = Add()([UP2, layer2_lateral])
    layer1_lateral = C_lateral(layer1, channel)
    UP3 = UpSampling2D((1, 2))(layer2_)
    layer1_ = Add()([UP3, layer1_lateral])

    # ----------PAN----------------------------------------------
    layer1_temp_condv = Conv2D(filters=channel,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               )(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = GlobalMaxPooling2D()(layer1_)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Conv2D(filters=channel,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='relu'
                      )(max_pool)

    max_pool = Activation('hard_sigmoid')(max_pool)
    layer1_mul = multiply([max_pool, layer1_temp_condv])

    layer1_b3 = Conv2D(filters=channel,
                       kernel_size=(1, 3),
                       strides=(1, 1),
                       padding='same'
                       )(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)

    layer1_mul = multiply([layer1_mul, result_1])

    layer2_up = Conv2D(filters=channel * 2,
                       kernel_size=(1, 2),
                       strides=(1, 2)
                       )(layer1_mul)

    layer2_pan_ = Conv2D(filters=channel * 2,
                         kernel_size=(1, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(layer2_)
    layer3_pan = Add()([layer2_up, layer2_pan_])

    layer3_pan_ = Conv2D(filters=channel,
                         kernel_size=(1, 3),
                         strides=(1, 2),
                         padding='same',
                         activation='relu',
                         )(layer3_pan)
    pan_Preoutput = Add()([layer3_pan_, layer3_])

    layer3_pan = Conv2D(filters=144, 
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=2
                        )(layer3_pan)
    layer3_pan = Conv2D(filters=96,  
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=4
                        )(layer3_pan)

    layer3_pan = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan)
    layer3_pan = Conv2D(filters=channel,
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu'
                        )(layer3_pan)

    pan_Preoutput = Add()([layer3_pan, pan_Preoutput]) 

    layer3_total_SepConv1 = SeparableConv2D(filters=192, kernel_size=(1, 2),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)
    layer3_total_SepConv2 = SeparableConv2D(filters=192, kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)

    Sep_Add = Add()([layer3_total_SepConv1, layer3_total_SepConv2]) 

    Dil_Add = Reshape((75, 192), )(Sep_Add)
    BiGru2 = Bidirectional(GRU(96, return_sequences=True))(Dil_Add)

    Gru2_output = Reshape((-1, 75, 192), )(BiGru2)
    Gru2_output = Conv2D(filters=96,
                         kernel_size=(1, 2),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(Gru2_output)

    layer3_total_SepConv1_1 = SeparableConv2D(filters=96, kernel_size=(1, 2),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    layer3_total_SepConv2_2 = SeparableConv2D(filters=96, kernel_size=(1, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)

    Sep_Add_1 = Add()([layer3_total_SepConv1_1, layer3_total_SepConv2_2]) 
    Sep_Add_1 = Conv2D(filters=96,
                       kernel_size=(1, 2),
                       strides=(1, 2),
                       padding='same',
                       activation='relu',
                       )(Sep_Add_1)

    Dil_Add_1 = Reshape((75, 96), )(Sep_Add_1)
    BiGru2_2 = Bidirectional(GRU(48, return_sequences=True))(Dil_Add_1)

    Gru2_output_1 = Reshape((-1, 75, 96), )(BiGru2_2)

    right_output = Add()([Gru2_output, Gru2_output_1])

    out1 = Concatenate()([right_output, pan_Preoutput])

    flat = Flatten(name='flatten')(out1)
    dense1 = Dense(84, activation='relu', name='dense1')(flat)
    # dense1 = Dropout(0.1)(dense1)
    dense1 = Dense(10, activation='relu', name='dense2')(dense1)
    d_out = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = tf.keras.backend.get_session()  # For Tensorflow 2

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    model_def.summary()

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def
# ------------------------------------------------

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))


def cnn_weights_loader(model_to_fill, cnn_appliance, pretrainedmodel_dir):
    log('Loading cnn weights from ' + cnn_appliance)
    weights_path = pretrainedmodel_dir + '/cnn_s2p_' + cnn_appliance + '_pointnet_model' + '_weights.h5'
    if not os.path.exists(weights_path):
        print('The directory does not exist or you do not have the files for trained model')

    f = h5py.File(weights_path, 'r')
    log(f.visititems(print_attrs))
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    for name in layer_names:
        if 'conv2d_' in name or 'cnn' in name:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]

            model_to_fill.layers[int(name[-1]) + 1].set_weights(weight_values)
            log('Loaded cnn layer: {}'.format(name))

    f.close()
    print('Model loaded.')


def weights_loader(model, path):
    log('Loading cnn weights from ' + path)
    model.load_weights(path + '_weights.h5')




def lightweight_mlp(x, in_channels=128, hidden_channels=64, out_channels=512):
    # 将输入 reshape 成四维以适应 DepthwiseConv2D
    x = tf.keras.layers.Reshape((x.shape[1], 1, x.shape[2]))(x)  # (batch_size, time_steps, 1, channels)
    x = tf.keras.layers.DepthwiseConv2D((1, 1), padding="same")(x)
    x = tf.keras.layers.LayerNormalization(axis=-1)(x)

    # 将输出还原为三维
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[-1]))(x)

    # 通道 MLP 模块
    x = tf.reduce_mean(x, axis=1)  # 在时间维度上求平均
    x = Dense(hidden_channels, activation='relu')(x)
    x = Dense(out_channels)(x)
    return x


def EVC(layer, window_length):
    # 轻量级 MLP 输出
    mlp_out = lightweight_mlp(layer)  # 输出形状为 (batch_size, out_channels)
    mlp_out = tf.keras.layers.Reshape((1, mlp_out.shape[-1]))(mlp_out)  # 扩展为 (batch_size, 1, out_channels)

    # 局部视觉中心机制
    lvc_out = Conv1D(128, 1, activation='relu', padding='same')(layer)
    lvc_out = GlobalAveragePooling1D()(lvc_out)  # 输出形状为 (batch_size, channels)
    lvc_out = tf.keras.layers.Reshape((1, lvc_out.shape[-1]))(lvc_out)

    # 拼接轻量级 MLP 和局部视觉中心输出
    evc_out = Concatenate(axis=-1)([mlp_out, lvc_out])  # (batch_size, 1, channels)

    # 上采样以接近 window_length
    upsample_size = max(1, window_length // evc_out.shape[1])  # 确保上采样比例为正整数
    evc_out = UpSampling1D(size=upsample_size)(evc_out)
    evc_out = Conv1D(256, 3, padding='same')(evc_out)  # 调整通道数

    # 使用 ZeroPadding1D 来调整长度到 window_length
    current_length = evc_out.shape[1]
    if current_length < window_length:
        padding_needed = window_length - current_length
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        evc_out = ZeroPadding1D(padding=(left_pad, right_pad))(evc_out)
    elif current_length > window_length:
        evc_out = tf.keras.layers.Cropping1D(cropping=(0, current_length - window_length))(evc_out)

    return evc_out



# 多尺度注意力机制
def multi_scale_attentionDepTemp(inputs, kernel_sizes=[7, 3]):

    Q = Conv1D(8, 1, padding="same")(inputs)
    K = Conv1D(8, 1, padding="same")(inputs)
    V = Conv1D(8, 1, padding="same")(inputs)

    # 聚合多尺度特征
    multi_scale_features = []
    for kernel_size in kernel_sizes:
        # 使用自定义 DepthwiseConv1D 实现多尺度特征提取
        Q_scaled = DepthwiseConv1D(kernel_size, padding="same")(Q)
        K_scaled = DepthwiseConv1D(kernel_size, padding="same")(K)
        V_scaled = DepthwiseConv1D(kernel_size, padding="same")(V)

        # 将每个通道的特征聚合回原始维度
        Q_scaled = Conv1D(8, 1, padding="same")(Q_scaled)
        K_scaled = Conv1D(8, 1, padding="same")(K_scaled)
        V_scaled = Conv1D(8, 1, padding="same")(V_scaled)

        # 计算注意力分数
        attention_scores = ReLU()(tf.matmul(Q_scaled, K_scaled, transpose_b=True))
        attention_scores = attention_scores / (tf.reduce_sum(attention_scores, axis=-1, keepdims=True) + 1e-6)

        # 生成注意力加权的 V
        weighted_V = tf.matmul(attention_scores, V_scaled)
        multi_scale_features.append(weighted_V)

    # 合并多尺度特征
    output = Concatenate(axis=-1)(multi_scale_features)
    output = Conv1D(inputs.shape[-1], 1, padding="same")(output)  # 输出维度映射回输入维度
    return output


def DepthwiseConv1D(kernel_size, padding="same", depth_multiplier=1):


    def expand_dim(x):
        return tf.expand_dims(x, axis=2)  # 在第二维度增加一维，高度变为 1

    def squeeze_dim(x):
        return tf.squeeze(x, axis=2)  # 移除第二维度，高度恢复为原来的 1D


    return tf.keras.Sequential([
        Lambda(expand_dim),  # 将输入扩展到 4D (batch_size, length, 1, channels)
        DepthwiseConv2D((kernel_size, 1), padding=padding, depth_multiplier=depth_multiplier),  # 进行 Depthwise 卷积
        Lambda(squeeze_dim)  # 将输出还原为 3D (batch_size, length, channels * depth_multiplier)
    ])

# -*- codeing = utf-8 -*-
# @Time : 2021/10/28 8:29
# @Author : Evan_wyl
# @File : xDeepFM.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings, random, math, os
from collections import namedtuple, OrderedDict

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal, glorot_uniform)
from tensorflow.python.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from RS.utils import SparseFeat, DenseFeat, VarLenSparseFeat

def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
    return dense_input_dict, sparse_input_dict

def build_embedding_layers(feature_columns, input_layer_dict, is_linear):
    embedding_layers_dict = dict()
    sparse_feature_column = list(filter(lambda x : isinstance(x, SparseFeat), feature_columns))

    if is_linear:
        for fc in sparse_feature_column:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_column:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)

    return embedding_layers_dict

def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    sparse_feature_column = list(filter(lambda x : isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for fc in sparse_feature_column:
        _input = input_layer_dict[fc.name]
        _embed = embedding_layer_dict[fc.name]
        embed = _embed(_input)

        if flatten:
            embed = Flatten()(embed)

        embedding_list.append(embed)
    return embedding_list


def get_dnn_output(dnn_input, hidden_units=[1024, 512, 256], dnn_dropout=0.3, activation='relu'):
    dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
    dropout = Dropout(dnn_dropout)

    x = dnn_input
    for dnn in dnn_network:
        x = dropout(dnn(x))

    return x

def get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns):
    concat_dense_logits = Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = Dense(1)(concat_dense_logits)

    linear_embedding_layers = build_embedding_layers(linear_feature_columns, sparse_input_dict, is_linear=True)

    sparse_1d_embed = []
    for fc in linear_feature_columns:
        if isinstance(fc, SparseFeat):
            feat_input = sparse_input_dict[fc.name]
            # Flatten层的主要作用是使维度对齐
            embed_ = Flatten()(linear_embedding_layers[fc.name](feat_input))
            sparse_1d_embed.append(embed_)

    # Add函数，把对应位置的数字相加
    sparse_logits_output = Add()(sparse_1d_embed)
    linear_part = Add()([dense_logits_output, sparse_logits_output])
    return linear_part

def activation_layer(activation):
    if isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation)
        )
    return act_layer

class CIN(Layer):
    def __init__(self, layer_size=(128,128), activation="relu", split_half=True, l2_reg=1e-5, seed=1024, **kwargs):
        if len(layer_size) == 0:
            raise ValueError("layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        # **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape))
            )
        self.feild_nums = [int(input_shape[1])]
        self.filters = []
        self.bias = []

        for i, size in enumerate(self.layer_size):
            self.filters.append(self.add_weight(name="filter" + str(i), shape=[1, self.feild_nums[-1] * self.feild_nums[0], size],
                                                dtype=tf.float32, initializer=glorot_uniform(seed=self.seed + i),
                                                regularizer=l2(self.l2_reg)))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number expect for the last layer when split_half=True"
                    )
                self.feild_nums.append(size // 2)
            else:
                self.feild_nums.append(size)

        self.activation_layers = [activation_layer(self.activation) for _ in self.layer_size]
        super(CIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs))
            )

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim, 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim, 2)

            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, self.feild_nums[0] * self.feild_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(dot_result, filters=self.filters[idx], stride=1, padding="VALID")
            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(curr_out)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1, keepdims=False)
        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self):
        config = {
            'layer_size':self.layer_size, 'split_half':self.split_half,
            'activation':self.activation, 'seed':self.seed
        }
        base_config = super(CIN, self).get_config()
        base_config.update(config)
        return base_config

def xDeepFM(linear_feature_columns, dnn_feature_columns, cin_size=[128, 128]):
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns)
    embedding_layer_dict = build_embedding_layers(linear_feature_columns+dnn_feature_columns, sparse_input_dict, is_linear=False)

    dnn_dense_feature_columns = list(filter(lambda  x : isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dnn_dense_feature_columns = [fc.name for fc in dnn_dense_feature_columns]
    dnn_concat_dense_inputs = Concatenate(axis=1)([dense_input_dict[col] for col in dnn_dense_feature_columns])

    dnn_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True)
    dnn_concat_sparse_kd_embed = Concatenate(axis=1)(dnn_sparse_kd_embed)

    dnn_input = Concatenate(axis=1)([dnn_concat_dense_inputs, dnn_concat_sparse_kd_embed])
    dnn_out = get_dnn_output(dnn_input)
    dnn_logits = Dense(1)(dnn_out)

    exFM_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=False)
    exFM_input = Concatenate(axis=1)(exFM_sparse_kd_embed)
    exFM_output = CIN(cin_size=cin_size)(exFM_input)
    exFM_logits = Dense(1)(exFM_output)

    stack_output = Add()([linear_logits, dnn_logits, exFM_logits])
    output_layer = Dense(1, activation="sigmoid")(stack_output)
    model = Model(input_layers, output_layer)
    return model



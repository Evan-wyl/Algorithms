# -*- codeing = utf-8 -*-
# @Time : 2021/10/28 8:28
# @Author : Evan_wyl
# @File : DeepFM_v1.py

import warnings
warnings.filterwarnings("ignore")
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from RS.utils import SparseFeat, DenseFeat, VarLenSparseFeat


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name)

    return dense_input_dict, sparse_input_dict

def build_embedding_layers(feature_column, input_layers_dict, is_linear):
    embedding_layers_dict = dict()

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_column)) if feature_column else []

    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)
    return embedding_layers_dict


def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_column):
    # Concatenate多个输入caoncat到一个输入
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = Dense(1)(concat_dense_inputs)
    linear_embedding_layers = build_embedding_layers(sparse_feature_column, sparse_input_dict, is_linear=True)

    sparse_1d_embed = []
    for fc in sparse_feature_column:
        feat_input = sparse_input_dict[fc.name]
        # 多维度输入一维度化
        embed = Flatten()(linear_embedding_layers[fc.name](feat_input))
        sparse_1d_embed.append(embed)

    sparse_logits_output = Add()(sparse_1d_embed)
    linear_logits = Add()[dense_logits_output, sparse_logits_output]
    return linear_logits


class FM_Layer(Layer):
    def __init__(self):
        super(FM_Layer, self).__init__()

    def call(self, inputs):
        concated_embed_values = inputs

        square_of_sum = tf.square(tf.reduce_sum(concated_embed_values, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(concated_embed_values * concated_embed_values, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

def get_fm_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns)) if sparse_feature_columns else []
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        _embed = dnn_embedding_layers[fc.name](feat_input)
        sparse_kd_embed.append(_embed)

    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    fm_cross_out = FM_Layer()(concat_sparse_kd_embed)
    return fm_cross_out


def get_dnn_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    sparse_feature_columns = list(filter(lambda x : isinstance(x, SparseFeat), sparse_feature_columns))

    sparse_kd_embed = []
    for fc in  sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        _embed = dnn_embedding_layers[fc.name](feat_input)
        _embed = Flatten()(_embed)
        sparse_kd_embed.append(_embed)

    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)

    mlp_out = Dropout(0.5)(Dense(256, activation="relu")(concat_sparse_kd_embed))
    mlp_out = Dropout(0.3)(Dense(256, activation="relu")(mlp_out))
    mlp_out = Dropout(0.1)(Dense(256, activation="relu")(mlp_out))

    dnn_out = Dense(1)(mlp_out)
    return dnn_out


def DeepFM(linear_feature_columns, dnn_feature_columns):
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)
    linear_sparse_feature_columns = list(filter(lambda x : isinstance(x, SparseFeat), linear_feature_columns))

    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)
    dnn_sparse_feature_columns = list(filter(lambda x : isinstance(x, SparseFeat), dnn_feature_columns))

    fm_logits = get_fm_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)
    dnn_logits = get_dnn_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)
    output_logits = Add()([linear_logits, fm_logits, dnn_logits])
    output_layers = Activation("sigmoid")(output_logits)
    model= Model(input_layers, output_layers)
    return model



# -*- codeing = utf-8 -*-
# @Time : 2021/10/28 12:08
# @Author : Evan_wyl
# @File : WideDeep_v1.py

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from RS.utils import SparseFeat, DenseFeat


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension), name=fc.name)

    return dense_input_dict, sparse_input_dict

def build_embedding_layers(feature_column, is_linear):
    embedding_layers_dict = dict()

    sparse_feature_column = list(filter(lambda x : isinstance(x, SparseFeat), feature_column))
    if is_linear:
        for fc in sparse_feature_column:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_column:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.dimension, name='kd_emb_' + fc.name)

    return embedding_layers_dict


def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_column):
    concat_dense_input = Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = Dense(1)(concat_dense_input)
    linear_embedding_layers = build_embedding_layers(sparse_feature_column, is_linear=True)

    sparse_1d_emb = []
    for fc in sparse_feature_column:
        feat_input = sparse_input_dict[fc.name]
        embed = Flatten()(linear_embedding_layers[fc.name](feat_input))
        sparse_1d_emb.append(embed)

    sparse_logits_output = Add()(sparse_1d_emb)
    linear_logits_output = Add()[dense_logits_output, sparse_logits_output]
    return linear_logits_output


def get_dnn_logits(sparse_input_dict, sparse_feature_columns):
    sparse_feature_columns = list(filter(lambda x : isinstance(x, SparseFeat), sparse_feature_columns))

    dnn_embedding_layers = build_embedding_layers(sparse_feature_columns, is_linear=False)

    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        embed_ = dnn_embedding_layers[fc.name](feat_input)
        embed_ = Flatten()(embed_)
        sparse_kd_embed.append(embed_)

    concat_sparse_kd_emb = Concatenate(axis=1)(sparse_kd_embed)

    mlp_out = Dropout(0.5)(Dense(256, activation="relu")(concat_sparse_kd_emb))
    mlp_out = Dropout(0.3)(Dense(256, activation="relu")(mlp_out))
    mlp_out = Dropout(0.1)(Dense(256, activation="relu")(mlp_out))

    dnn_out = Dense(1)(mlp_out)
    return dnn_out



def WideDeepModel(linear_feature_columns, dnn_feature_columns):
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)
    linear_sparse_feature_columns = list(filter(lambda x : isinstance(x, linear_feature_columns), linear_feature_columns))
    linear_logits = get_linear_logits(dense_input_dict, linear_feature_columns, linear_sparse_feature_columns)
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    dnn_logits = get_dnn_logits(sparse_input_dict, dnn_feature_columns)
    concat_result = Concatenate(axis=1)[linear_logits, dnn_logits]
    output = Dense(1, activation="sigmoid")(concat_result)
    model = Model(input_layers, output)
    return model





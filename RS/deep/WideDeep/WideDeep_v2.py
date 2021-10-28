# -*- codeing = utf-8 -*-
# @Time : 2021/10/28 12:09
# @Author : Evan_wyl
# @File : WideDeep_v2.py

from tensorflow import keras

# keras.experimental公共命名空间的API
linear_model = keras.experimental.LinearModel()
dnn_model = keras.Sequential([keras.layers.Dense(units=64),
                              keras.layers.Dense(units=1)])
combined_model = keras.experimental.WideDeepModel(linear_model, dnn_model)

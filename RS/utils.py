# -*- codeing = utf-8 -*-
# @Time : 2021/10/28 10:42
# @Author : Evan_wyl
# @File : utils.py

from collections import namedtuple

SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vaocabulary_size', 'embedding_dim', 'maxLen'])

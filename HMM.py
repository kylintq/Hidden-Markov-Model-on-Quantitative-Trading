# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:53:05 2020

@author: TQ
"""

import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3",'box4']
n_states = len(states)
#观测状态
observations = ["red", "white"]
n_observations = len(observations)
model = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
#观测结果
D1 = [[1], [0], [0], [0], [1], [1], [1]]
D2 = [[1], [0], [0], [0], [1], [1], [1], [0], [1], [1]]
D3 = [[1], [0], [0]]
D4=[[1],[0],[1],[1],[0]]

X = np.concatenate([D1, D2, D3,D4])


model.fit(X)

print (model.transmat_)
'''
class hmmlearn.hmm.GaussianHMM(n_components=1, covariance_type='diag',
                               min_covar=0.001, startprob_prior=1.0,
                               transmat_prior=1.0, means_prior=0, 
                               means_weight=0, covars_prior=0.01,
                               covars_weight=1, algorithm='viterbi',
                               random_state=None, n_iter=10, tol=0.01,
                               verbose=False, params='stmc', init_params='stmc')
n_components : 隐藏状态数目
covariance_type: 协方差矩阵的类型
min_covar : 最小方差，防止过拟合
startprob_prior : 初始概率向量
transmat_prior : 转移状态矩阵
means_prior, means_weight : 均值
covars_prior, covars_weight : 协方差
algorithm : 所用算法
random_state : 随机数种子
n_iter : 最大迭代次数
tol : 停机阈值
verbose : 是否打印日志以观察是否已收敛
params : 决定哪些参数在迭代中更新
init_params : 决定哪些参数在迭代前先初始化
'''
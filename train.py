

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tl2cgen
import treelite

def calculate_metrics(true_labels, predicted_labels):
    """
    计算召回率和精确率。

    参数:
    true_labels : np.ndarray
        真实标签的布尔数组，True为正类。
    predicted_labels : np.ndarray
        预测标签的布尔数组，True为正类。

    返回:
    tuple
        包含召回率和精确率的元组。
    """
    # 确保两个数组长度相同
    assert len(true_labels) == len(predicted_labels), "两个数组长度必须相等"

    # 计算TP, FP, FN
    TP = np.sum((predicted_labels == True) & (true_labels == True)) / len(true_labels)
    FP = np.sum((predicted_labels == True) & (true_labels == False)) / len(true_labels)
    FN = np.sum((predicted_labels == False) & (true_labels == True)) / len(true_labels)
    TN = np.sum((predicted_labels == False) & (true_labels == False)) / len(true_labels)

    # 计算召回率和精确率，考虑避免除以零的情况
    # recall = (TP + TN) / len(true_labels) if TP + FN != 0 else 0
    # precision = TP / (TP + FP) if TP + FP != 0 else 0

    return TP, FP, FN, TN


if __name__ == "__main__":
    # 加载数据集
    data = pd.read_csv("/home/tianlan.lht/code/github/github-vsag/data.csv").to_numpy(dtype="float32")
    # test = pd.read_csv("/home/tianlan.lht/code/github/github-vsag/test.csv").to_numpy(dtype='float32')
    X = data[:,:-1]
    y = data[:,-1]
    # X_gt = test[:,:-1]
    # y_gt = test[:,-1]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=73)

    # 转换为LightGBM的数据格式
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 定义模型参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'lambda_l1': 1e-4,
        'lambda_l2': 1e-4,
        'verbose': 1
    }

    model = lgb.Booster(model_file="model.txt")
    # 训练模型
    # model = lgb.train(params, train_data, num_boost_round=15)

    # 预测
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    TP, FP, FN, TN = calculate_metrics(y_test, y_pred > 0.45)

    print(TP, FP, FN, TN)
    #
    # y2 = model.predict(X_gt, num_iteration=model.best_iteration)
    #
    # TP, FP, FN, TN = calculate_metrics(y_gt, y2 > 0.4)

    print(TP, FP, FN, TN)

    # 特征重要性
    feature_importance = model.feature_importance()

    # 打印特征重要性
    print(feature_importance)

    model = treelite.Model.load('model2.txt', model_format='lightgbm')

    tl2cgen.generate_c_code(model=model, dirpath="./t2l", params={})
    tl2cgen.generate_cmakelists("./t2l")
    # model.save_model("model2.txt")

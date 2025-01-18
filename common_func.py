# common_func.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, matthews_corrcoef, precision_score, \
    recall_score, f1_score

# 数据读取函数
def read_data(file_path):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return x, y

# 训练过程中的损失历史记录类
class LossHistory(Callback):  # 继承自 Callback
    def __init__(self):
        super(LossHistory, self).__init__()  # 调用父类的构造函数
        self.val_accuracies = []
        self.accuracies = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accuracies.append(logs.get('accuracy'))
        self.val_accuracies.append(logs.get('val_accuracy'))

    def loss_plot(self, save_path=None):
        plt.figure()
        plt.plot(self.losses, label='train loss')
        plt.plot(self.val_losses, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss during Training')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# 评估方法
class evaluate_method:
    @staticmethod
    def get_acc(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def get_kappa(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return cohen_kappa_score(y_true, y_pred)

    @staticmethod
    def get_IOA(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return np.mean(y_true == y_pred)

    @staticmethod
    def get_mcc(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def get_recall(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return recall_score(y_true, y_pred)

    @staticmethod
    def get_precision(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return precision_score(y_true, y_pred)

    @staticmethod
    def get_f1(y_true, y_pred_prob):
        y_pred = np.round(y_pred_prob).astype(int)
        return f1_score(y_true, y_pred)

    @staticmethod
    def get_ROC(y_true, y_pred_prob, save_path=None):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        if save_path:
            plt.savefig(save_path)
        plt.show()

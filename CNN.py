import pandas as pd
from sklearn import metrics
from keras.utils import np_utils
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, Add, BatchNormalization
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 设置随机种子以便于结果重现
tf.random.set_seed(6)
np.random.seed(6)


# 读取数据
def read_data(file_path):
    df = pd.read_csv(file_path)
    features = df[['NDVI_MEAN', 'DEM_ADJ', 'ROUGH_MEAN', 'SLOPE_MEAN', 'SLOPE_VAR', 'PLANCURV', 'POU_WAM', 'R_Index']]
    labels = df['is_prototype'].values
    return features.values, labels


train_x, train_y_1D = read_data('training_samples.csv')
test_x, test_y_1D = read_data('test_samples.csv')

# 转换标签为分类格式
num_classes = 2
train_y = np_utils.to_categorical(train_y_1D, num_classes)
test_y = np_utils.to_categorical(test_y_1D, num_classes)

# 扩展维度以适应CNN输入
train_x = np.expand_dims(train_x, axis=2)
test_x = np.expand_dims(test_x, axis=2)


# 残差块定义
def residual_block(x, filters):
    shortcut = x
    x = Conv1D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


# 定义ResNet模型
input_layer = Input(shape=(train_x.shape[1], 1))
x = Conv1D(32, kernel_size=3, padding='same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)

for _ in range(3):
    x = residual_block(x, 32)

x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.7)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
optimizer = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 打印模型摘要
print(model.summary())

# 保存最佳模型的回调
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 训练模型
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=2,
                    callbacks=[checkpoint, early_stopping], batch_size=64, epochs=100)

# 输出最佳验证准确率
best_val_accuracy = max(history.history['val_accuracy'])
print("Best Validation Accuracy = ", best_val_accuracy)

# 加载最佳模型
model.load_weights('best_model.h5')

# 预测
y_prob_test = model.predict(test_x)
y_pred_test = np.argmax(y_prob_test, axis=1)
y_true_test = np.argmax(test_y, axis=1)
y_probability_first = [prob[1] for prob in y_prob_test]

# 计算AUC
test_auc = metrics.roc_auc_score(test_y_1D, y_probability_first)
print("AUC = ", test_auc)

# 计算测试集准确率
test_accuracy = metrics.accuracy_score(y_true_test, y_pred_test)
print("Test Accuracy = ", test_accuracy)


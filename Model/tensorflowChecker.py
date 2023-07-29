# # Checker1: Mnist checker
# import tensorflow as tf
# # 载入MINST数据集
# mnist = tf.keras.datasets.mnist
# # 划分训练集和测试集
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


# # 定义模型结构和模型参数
# model = tf.keras.models.Sequential([
#     # 输入层28*28维矩阵
#     tf.keras.layers.Flatten(input_shape=(28, 28)), 
#     # 128维隐层，使用relu作为激活函数
#     tf.keras.layers.Dense(128, activation='relu'), 
#     tf.keras.layers.Dropout(0.2),
#     # 输出层采用softmax模型，处理多分类问题
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# # 定义模型的优化方法(adam)，损失函数(sparse_categorical_crossentropy)和评估指标(accuracy)
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# # 训练模型，进行5轮迭代更新(epochs=5）
# model.fit(x_train, y_train, epochs=5)
# # 评估模型
# model.evaluate(x_test,  y_test, verbose=2)

# Checker2: cifar 100 Checker
# import tensorflow as tf

# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64)
# model.evaluate(x_test,  y_test, verbose=2)

# 检查GPU是否可用
import sys
import tensorflow
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
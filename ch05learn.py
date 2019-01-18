import sys, os
sys.path.append(os.pardir)
import numpy as np
import load_mnist_numbers
from ch05 import TwoLayerNet

# データの読み込み
(train_data, train_label), (verify_data, verify_label) = load_mnist_numbers.load_mnist(normalize_flag_=True, flat_flag_=True, one_hot_flag_=True)

network = TwoLayerNet(input_size_=784, hidden_size_=50, output_size_=10)

iters_num = 10000
train_size = train_data.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    train_data_batch = train_data[batch_mask]
    train_label_batch = train_label[batch_mask]

    # 誤差逆伝搬法によって勾配を求める
    grad = network.gradient(train_data_batch, train_label_batch)

    # 更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(train_data_batch, train_label_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(train_data, train_label)
        test_acc = network.accuracy(verify_data, verify_label)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

print("finished!?")

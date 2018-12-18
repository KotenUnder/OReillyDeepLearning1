import numpy as np
import load_mnist_numbers

# ステップ関数
# 引数が正なら1を返す、０か負なら0を返します。
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_array(x):
    y = x > 0
    return y.astype(np.int)

# シグモイド関数
# x=0で1/2を返し、ゆるやかに0から1へ返す値が変化する
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU関数(Rectified Linear Unit)
# 入力が負の場合は０を、非負の場合は入力値をそのまま返します
def ReLU(x):
    return np.maximum(0, x)

# 恒等関数
# 常に入力と同じ値を返します
def identity_function(x):
    return x


# ソフトマックス関数
# exp(a_k) / sum (exp(a_i) )
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

if __name__ == '__main__':
    print(step_function_array(np.array([1,2,3])))
    train_data, train_label = load_mnist_numbers.load_train()
    verify_data, verify_label = load_mnist_numbers.load_verify()


    print("done")



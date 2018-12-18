import os
import numpy as np
import pickle

# 定数
DATA_MAGIC_NUMBER = 2051
LABEL_MAGIC_NUMBER = 2049

# データファイルの場所
PWD = os.path.dirname(__file__)
DATA_FOLDER_NAME = os.path.join(PWD, "data", "mnist_number")
DATA_TRAIN_IMAGES = os.path.join(DATA_FOLDER_NAME, "train-images.idx3-ubyte")
DATA_TRAIN_LABELS = os.path.join(DATA_FOLDER_NAME, "train-labels.idx1-ubyte")
DATA_VERIFY_IMAGES = os.path.join(DATA_FOLDER_NAME, "t10k-images.idx3-ubyte")
DATA_VERIFY_LABELS = os.path.join(DATA_FOLDER_NAME, "t10k-labels.idx1-ubyte")

# データピクルスの保存場所
PICKLE_TRAIN_IMAGES = os.path.join(DATA_FOLDER_NAME, "mnist_number_train_images.pickle")
PICKLE_TRAIN_LABELS = os.path.join(DATA_FOLDER_NAME, "mnist_number_train_labels.pickle")
PICKLE_VERIFY_IMAGES = os.path.join(DATA_FOLDER_NAME, "mnist_number_verify_images.pickle")
PICKLE_VERIFY_LABELS = os.path.join(DATA_FOLDER_NAME, "mnist_number_verify_labels.pickle")

def load_mnist(flat_flag_ = False, normalize_flag_ = False, one_hot_flag_ = False):
    trains = _read_data_and_label(DATA_TRAIN_IMAGES, DATA_TRAIN_LABELS, flat_flag_, normalize_flag_, one_hot_flag_)
    verifies = _read_data_and_label(DATA_VERIFY_IMAGES, DATA_VERIFY_LABELS, flat_flag_, normalize_flag_, one_hot_flag_)

    return trains, verifies


def load_train():
    return _read_data_and_label(DATA_TRAIN_IMAGES, DATA_TRAIN_LABELS, False, False, True)

def load_verify():
    return _read_data_and_label(DATA_VERIFY_IMAGES, DATA_VERIFY_LABELS, False, False, True)

def load_train_():
    if (os.path.isfile(PICKLE_TRAIN_IMAGES) and os.path.isfile(PICKLE_TRAIN_LABELS)):
        data = _pickle_load(PICKLE_TRAIN_IMAGES)
        label = _pickle_load(PICKLE_TRAIN_LABELS)
    else:
        data, label = _read_data_and_label(DATA_TRAIN_IMAGES, DATA_TRAIN_LABELS)
        _pickle_object(data, PICKLE_TRAIN_IMAGES)
        _pickle_object(label, PICKLE_TRAIN_LABELS)

    return data, label


def load_verify_():
    if (os.path.isfile(PICKLE_VERIFY_IMAGES) and os.path.isfile(PICKLE_VERIFY_LABELS)):
        data = _pickle_load(PICKLE_VERIFY_IMAGES)
        label = _pickle_load(PICKLE_VERIFY_LABELS)
    else:
        data, label = _read_data_and_label(DATA_VERIFY_IMAGES, DATA_VERIFY_LABELS)
        _pickle_object(data, PICKLE_VERIFY_IMAGES)
        _pickle_object(label, PICKLE_VERIFY_LABELS)

    return data, label


def _read_data_and_label(data_path_, label_path_, flat_array_=False, normalize_pixel_=False, one_hot_=False):
    train_data_stream = open(data_path_, "rb")
    train_label_stream = open(label_path_, "rb")

    # 画像の数やサイズなどのヘッダを読み込む
    data_header = np.fromfile(train_data_stream, ">I", 4)
    label_header = np.fromfile(train_label_stream, ">I", 2)

    # マジックナンバーのチェック
    if (data_header[0] != DATA_MAGIC_NUMBER or label_header[0] != LABEL_MAGIC_NUMBER):
        raise Exception("ファイルフォーマットが違います。")
    # 画像の数、サイズの設定
    image_num = data_header[1]
    if (image_num != label_header[1]):
        raise Exception("dataとtrainで画像の数が違います。")

    image_column = data_header[2]
    image_row = data_header[3]

    # データ本体の読み込み
    image_pixel = image_column * image_row
    # 出力用変数
    data = []
    try:
        # imageに関する読み込み処理
        for i in range(0, image_num):
            data_oneimage = np.fromfile(train_data_stream, np.uint8, image_pixel)
            # normalize有効時は0～255を0～1に正規化する
            if normalize_pixel_:
                pass

            # flat有効時はそのまま、flat無効時はcolumn * rowにreshapeする
            if flat_array_:
                data.append(data_oneimage)
            else:
                data.append(data_oneimage.reshape(image_column, image_row))

        # labelsに関する読み込み処理
        label = np.fromfile(train_label_stream, np.uint8, -1)
        # hotspot形式がTrueの場合は正解のみ1の配列リストとする。
        if one_hot_:
            label_list = label
            label = []
            for case in label_list:
                base = [0] * 10
                base[case] = 1
                label.append(base)

    except:
        raise Exception("ファイルフォーマットが違います。")

    return data, label


def _pickle_object(pickle_object_, pickle_object_path_):
    with open(pickle_object_path_, "wb") as f:
        pickle.dump(pickle_object_, f)


def _pickle_load(pickle_object_path_):
    with open(pickle_object_path_, "rb") as f:
        obj = pickle.load(f)
        return obj

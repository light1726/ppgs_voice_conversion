import tensorflow as tf
import numpy as np
import pickle
from random import sample
from scipy.interpolate import interp1d
import os

DATA_DIR = '/data/data/vc_data/zhiling'
LF0_MEAN_F = '/data/data/vc_data/zhiling/statistics/lf0/mean.npy'
LF0_STD_F = '/data/data/vc_data/zhiling/statistics/lf0/std.npy'
LPC_MEAN_F = '/data/data/vc_data/zhiling/statistics/lpc/mean.npy'
LPC_STD_F = '/data/data/vc_data/zhiling/statistics/lpc/std.npy'
LC_DIM = 346
NUM_MEL = 37


# STFT_DIM = 513


def sorted_file_list(data_dir, pattern='.lf0'):
    # default use lf0 data directory
    file_list = [(f.split('.')[0], os.path.getsize(os.path.join(data_dir, f)))
                 for f in os.listdir(data_dir) if f.endswith(pattern)]
    file_list.sort(key=lambda s: s[1])
    sorted_id_list = [f for f, _ in file_list]
    return sorted_id_list


def split_dataset(id_list, dev_size=50):
    # split dataset into train and dev set
    dev_set = sample(id_list, dev_size)
    for item in dev_set:
        id_list.remove(item)
    datasplit = {'train': id_list, 'dev': dev_set}
    with open('data_split.pkl', 'wb') as f:
        pickle.dump(datasplit, f, protocol=pickle.HIGHEST_PROTOCOL)
    return datasplit


def get_files_list(data_dir, mode='train'):
    # mode: 'train' --> get training data;
    #       'dev'   --> get dev data;
    #       'all'   --> get all ( both train and dev) data, used for generation.
    assert mode in ['train', 'dev', 'all'] and 'mode should be one of aforementioned'
    ppgs_dir = os.path.join(data_dir, 'ppgs960')
    lf0_dir = os.path.join(data_dir, 'lf0_10ms')
    lpc_dir = os.path.join(data_dir, 'lpcnet_npy')
    # read or create data split dictionary
    data_split_f = 'data_split.pkl'
    if not os.path.isfile(data_split_f):
        fid_list = sorted_file_list(lf0_dir, pattern='.lf0')
        data_dict = split_dataset(fid_list, dev_size=50)
    else:
        with open(data_split_f, 'rb') as f:
            data_dict = pickle.load(f)
    if mode == 'all':
        ppgs_files = ([os.path.join(ppgs_dir, f + '.npy') for f in data_dict['train']] +
                      [os.path.join(ppgs_dir, f + '.npy') for f in data_dict['dev']])
        lf0_files = ([os.path.join(lf0_dir, f + '.lf0') for f in data_dict['train']] +
                     [os.path.join(lf0_dir, f + '.lf0') for f in data_dict['dev']])
        lpc_files = ([os.path.join(lpc_dir, f + '.npy') for f in data_dict['train']] +
                     [os.path.join(lpc_dir, f + '.npy') for f in data_dict['dev']])
    else:
        ppgs_files = [os.path.join(ppgs_dir, f + '.npy') for f in data_dict[mode]]
        lf0_files = [os.path.join(lf0_dir, f + '.lf0') for f in data_dict[mode]]
        lpc_files = [os.path.join(lpc_dir, f + '.npy') for f in data_dict[mode]]
    return ppgs_files, lf0_files, lpc_files


def all_data_generator():
    ppg_files, lf0_files, stft_files = get_files_list(DATA_DIR, mode='all')
    for ppg_f, lf0_f, lpc_f in zip(ppg_files, lf0_files, stft_files):
        fname = lpc_f.split('/')[-1]
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        lpc = np.load(lpc_f)
        ppg = _softmax(ppg)
        lf0 = lf0_normailze(lf0, mean_f=LF0_MEAN_F, std_f=LF0_STD_F)
        lpc = lpcnet_feats_normalize(lpc, mean_f=LPC_MEAN_F, std_f=LPC_STD_F)
        ppg, lf0, lpc = normalize_length(ppg, lf0, lpc)
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, lc.shape[0], lpc, fname


def train_generator():
    ppg_files, lf0_files, lpc_files = get_files_list(DATA_DIR, mode='train')
    for ppg_f, lf0_f, lpc_f in zip(ppg_files, lf0_files, lpc_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        lpc = np.load(lpc_f)
        ppg = _softmax(ppg)
        lf0 = lf0_normailze(lf0, mean_f=LF0_MEAN_F, std_f=LF0_STD_F)
        lpc = lpcnet_feats_normalize(lpc, mean_f=LPC_MEAN_F, std_f=LPC_STD_F)
        ppg, lf0, lpc = normalize_length(ppg, lf0, lpc)
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, lc.shape[0], lpc


def dev_generator():
    ppg_files, lf0_files, lpc_files = get_files_list(DATA_DIR, mode='dev')
    for ppg_f, lf0_f, lpc_f in zip(ppg_files, lf0_files, lpc_files):
        ppg = np.load(ppg_f)
        lf0 = np.fromfile(lf0_f, dtype=np.float32)
        lpc = np.load(lpc_f)
        ppg = _softmax(ppg)
        lpc = lpcnet_feats_normalize(lpc)
        lf0 = lf0_normailze(lf0)
        ppg, lf0, lpc = normalize_length(ppg, lf0, lpc)
        lc = np.concatenate([ppg, lf0[:, None]], axis=-1)
        yield lc, lc.shape[0], lpc


def _softmax(x, axis=-1):
    assert len(x.shape) == 2
    _max = np.max(x)
    probs = np.exp(x - _max) / np.sum(np.exp(x - _max), axis=axis, keepdims=True)
    return probs


def lpcnet_feats_normalize(lpc_feats, mean_f=None, std_f=None):
    _mean = np.load(mean_f) if mean_f is not None else np.mean(lpc_feats, axis=0)
    _std = np.load(std_f) if std_f is not None else np.std(lpc_feats, axis=0)
    normalized = (lpc_feats - _mean) / _std
    return normalized


def lf0_normailze(lf0, mean_f=None, std_f=None):
    mean = np.load(mean_f) if mean_f is not None else np.mean(lf0[lf0 > 0])
    std = np.load(std_f) if std_f is not None else np.std(lf0[lf0 > 0])
    normalized = np.copy(lf0)
    normalized[normalized > 0] = (lf0[lf0 > 0] - mean) / std
    normalized[0] = 1e-5 if normalized[0] <= 0 else normalized[0]
    normalized[-1] = 1e-5 if normalized[-1] <= 0 else normalized[-1]
    non_zero_ids = np.where(normalized > 0)[0]
    non_zero_vals = normalized[non_zero_ids]
    f = interp1d(non_zero_ids.astype(np.float32), non_zero_vals)
    x_all = np.arange(len(normalized), dtype=np.float32)
    interpolated = f(x_all)
    return interpolated


def normalize_length(ppg, lf0, lpc):
    min_len = min([ppg.shape[0], len(lf0), lpc.shape[0]])
    ppg = ppg[: min_len, :]
    lf0 = lf0[: min_len]
    lpc = lpc[: min_len, :]
    return ppg, lf0, lpc


def process(item1, item2, item3):
    return [{'ppg_lf0_inputs': item1,
            'sequence_length': item2},
            {'outputs': item3}]


def dataset_test():
    batch_size = 8
    train_set = tf.data.Dataset.from_generator(
        train_generator,
        output_types=(tf.float32, tf.int32, tf.float32),
        output_shapes=([None, LC_DIM], [], [None, NUM_MEL]))
    train_set = train_set.padded_batch(batch_size,
                                       ([None, LC_DIM], [],
                                        [None, NUM_MEL]))

    dev_set = tf.data.Dataset.from_generator(
        dev_generator,
        output_types=(tf.float32, tf.int32, tf.float32),
        output_shapes=([None, LC_DIM], [], [None, NUM_MEL]))

    new_dev_set = dev_set.map(process)
    for elem in new_dev_set:
        print(elem)


if __name__ == '__main__':
    dataset_test()

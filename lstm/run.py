import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

dataset = "."  # 数据集
# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'  # 词向量
model_name = 'TextRNN'  # TextCNN, TextRNN, FastText
word = False # True for word, False for char

if model_name == 'FastText':
    from utils_fasttext import build_dataset, build_iterator, get_time_dif
    embedding = 'random'
else:
    from utils import build_dataset, build_iterator, get_time_dif

x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)

# 设置随机种子
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
vocab, train_data, dev_data, test_data = build_dataset(config, word)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
config.n_vocab = len(vocab)
model = x.Model(config).to(config.device)
writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
if model_name != 'Transformer':
    init_network(model)
print(model.parameters)
train(config, model, train_iter, dev_iter, test_iter,writer)
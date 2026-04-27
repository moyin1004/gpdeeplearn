import medmnist
print(medmnist.__version__)

import os

import model
from medmnist.dataset import INFO
from medmnist.evaluator import getAUC, getACC, save_results
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


batch_size = 64
lr = 0.001
start_epoch = 0
end_epoch = 1

data_flag = "pathmnist"
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
print(info)

DataClass = getattr(medmnist, info['python_class'])

print(DataClass)

# 获取数据
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(root="data", split='train', transform=train_transform, download=True)
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DataClass(root="data", split='val', transform=val_transform, download=True)
val_loader = data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DataClass(root="data", split='test', transform=test_transform, download=True)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True)

print('==> Building and training model...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = model.ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)

print(task)
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


def val(net, val_loader, device, val_auc_list, task, dir_path, epoch):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    '''

    net.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = net(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': net.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)

val_auc_list = []
dir_path = os.path.join("data", '%s_checkpoints' % (data_flag))
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for epoch in trange(start_epoch, end_epoch):
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            # 去掉维度为1的维
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    val(net, val_loader, device, val_auc_list, task, dir_path, epoch)

import numpy as np
auc_list = np.array(val_auc_list)
index = auc_list.argmax()
print('epoch %s is the best model' % (index))

print('==> Testing model...')

def test(net, split, data_loader, device, flag, task, output_root=None):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    net.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = net(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)

output_root = "data"

restore_model_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
net.load_state_dict(torch.load(restore_model_path)['net'])
test(net, 'train', train_loader, device, data_flag, task, output_root=output_root)
test(net, 'val', val_loader, device, data_flag, task, output_root=output_root)
test(net, 'test', test_loader, device, data_flag, task, output_root=output_root)
torch.cuda.empty_cache()

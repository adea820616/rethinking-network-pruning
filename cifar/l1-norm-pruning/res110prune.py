import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=110,
                    help='depth of the resnet')
parser.add_argument('--model', default='/home/jovyan/model_compression/rethinking-network-pruning/cifar/l1-norm-pruning/logs/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='/home/jovyan/model_compression/rethinking-network-pruning/cifar/l1-norm-pruning/logs/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('-v', default='B', type=str, 
                    help='version of the model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)

if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    '''
    load test data
    比對prediction and answer
    '''
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)

skip = {
    'A': [36],
    'B': [36, 38, 74],
}

prune_prob = {
    'A': [0.5, 0.0, 0.0],
    'B': [0.5, 0.4, 0.3],
}

layer_id = 1
cfg = []
cfg_mask = []
#記錄誰要留下誰不要留下
for m in model.modules():
    '''
    ResNet(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    '''
    if isinstance(m, nn.Conv2d):
        '''
        - m:
            Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        - m.weight.data.shape[0]:
            torch.Size([16, 3, 3, 3])
            torch.Size([16, 16, 3, 3])
        - out_channels:
            16
            16
        '''
        out_channels = m.weight.data.shape[0]
        
        #如果是skip layer
        if layer_id in skip[args.v]: #args.v: A or B
            #[tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])...]
            cfg_mask.append(torch.ones(out_channels)) #一個空的16維(output_channels size) 裡面都存1的matrix

            #[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
            cfg.append(out_channels) #12個?
            layer_id += 1
            continue
        #如果不是skip layer就剪第2個layer
        if layer_id % 2 == 0: #第2個layer
            stage = layer_id // 36 #看現在這個layer是0 1 2 哪個stage(一個stage 18 Block/36 layer)
            if layer_id <= 36:
                stage = 0
            elif layer_id <= 72:
                stage = 1
            elif layer_id <= 108:
                stage = 2
            prune_prob_stage = prune_prob[args.v][stage] # A: 0.5/0/0 ; B: 0.5/0.4/0.3

            #.clone(): 複製一個完全相同的tensor
            #.cpu(): gpu tensor 轉 cpu tensor
            #.numpy(): tensor 轉 numpy
            # k 3*3, 3channel, output channel 16
            weight_copy = m.weight.data.abs().clone().cpu().numpy() #weight 絕對值
            L1_norm = np.sum(weight_copy, axis=(1,2,3)) # 算L1 全部加總， 3*3*3*16 -> 1*16
            num_keep = int(out_channels * (1 - prune_prob_stage)) # 要留下的數量 16 * (1-0.5) = 8
            arg_max = np.argsort(L1_norm) #[11  0  8 15  1 12  9  6 14  3 13  4  7 10  5  2]
            #x中的元素從小到大排列，提取其對應的index(索引)，然後輸出到y
            # x=np.array([1,4,3,-1,6,9])
            # y=array([3,0,2,1,4,5])
            
            #[::-1]從後面取回來，[:num_keep]取要留下的8個
            arg_max_rev = arg_max[::-1][:num_keep] #[ 2  5 10  7  4 13  3 14]
            mask = torch.zeros(out_channels) #一個空的 裡面都存0的matrix， 1*16維

            #把要留下的人給1
            mask[arg_max_rev.tolist()] = 1 #tensor([0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.])
            cfg_mask.append(mask) #tensor([0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.], tensor([])....)
            cfg.append(num_keep) #[8, 8, ...]
            layer_id += 1
            continue
        layer_id += 1

newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3) #tensor([1., 1., 1.])
layer_id_in_cfg = 0
conv_count = 1
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    # CONV
    if isinstance(m0, nn.Conv2d):
        '''conv1'''
        #第一個conv layer 1 weight 複製
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        
        '''
        layer1.0.conv1 (stage 1, Block 0, conv 1)
        layer1.1.conv1 (stage 1, Block 1, conv 1)
        layer1.2.conv1 (stage 1, Block 2, conv 1)
        '''
        # block 內的 conv layer = 2 
        if conv_count % 2 == 0:
            #cfg_mask: tensor([0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.], tensor([])....)
            mask = cfg_mask[layer_id_in_cfg] #選第一個cfg: tensor([0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.])
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy()))) #值為非0的，也就是等於1: [ 2  3  4  5  7 10 13 14]
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone() #C 留下[ 2  3  4  5  7 10 13 14]  weight就會有 3*3, 3 channels, 8output_channels
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1 #+
            conv_count += 1
            continue

        '''
        layer1.0.conv2 (stage 1, Block 0, conv 2)
        layer1.1.conv2 (stage 1, Block 1, conv 2)
        layer1.2.conv2 (stage 1, Block 2, conv 2)
        '''
        # block 內的 conv layer = 1
        if conv_count % 2 == 1:
            #cfg_mask: tensor([0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.], tensor([])....)
            mask = cfg_mask[layer_id_in_cfg-1] #拿跟上面一樣的mask
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone() #取conv 1在conv2會被剪掉的地方
            m1.weight.data = w.clone()
            conv_count += 1
            continue
    # Batch
    elif isinstance(m0, nn.BatchNorm2d):
        #也剪相對應的batch
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            # BN公式: Y = (X - running_mean) / sqrt(running_var + eps) * gamma + beta
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    #Linear
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
model = newmodel
acc = test(model)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
print("number of parameters: "+str(num_parameters))
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test Accuracy: \n"+str(acc)+"\n")
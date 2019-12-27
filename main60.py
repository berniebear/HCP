import torch
from torch.utils.data.dataset import Dataset
import os, argparse, json, time
from torch.autograd import Variable
import numpy as np
import random
from utils import *
from model import *
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='')
parser.add_argument('--kfold', default=10, type=int, help='kfold')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') # 0.000001
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=90, type=int, help='batch size')
parser.add_argument('--loss', default='l1', type=str, help='loss')
parser.add_argument('--act', default='leaky_relu', type=str, help='actication')
parser.add_argument('--layers', default=2, type=int, help='layers')
parser.add_argument('--step', default=60, type=int, help='step size to shrink lr')
parser.add_argument('--optim', default='adam', help='activation')
parser.add_argument('--model_name', default='model0', type=str, help='model_name')
parser.add_argument('--hidden', default=128, type=int, help='hidden size')
parser.add_argument('--nval', default=8, type=int, help='hidden size')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--save_every', default=30, type=int, help='save every n epoch')
args = parser.parse_args()
torch.manual_seed(args.seed) 
np.random.seed(args.seed)
random.seed(args.seed)

def main(args):
    model_dir = os.path.join('models/', args.model_name)
    log_dir = os.path.join('logs', args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # test_data
    data = np.load('data/subjects.npy')
    labels = 2.5*(np.load('data/neofac.npy')-0.3) # normalize to [0,1]
    splits = get_kfold(np.arange(data.shape[0]), k=args.kfold)

    trainval_lst, test_lst = splits[0] # we sample only the fist fold, repeat 1000 times
    # use full trainval as train
    train_lst = trainval_lst
    #train_lst = trainval_lst[:-args.nval]
    # use 36 samples as validation
    val_lst = trainval_lst[-args.nval:]

    tr_data=torch.utils.data.TensorDataset(torch.FloatTensor(data[train_lst]).cuda(), torch.FloatTensor(labels[train_lst]).cuda())
    tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=False)
    val_data=torch.utils.data.TensorDataset(torch.FloatTensor(data[val_lst]).cuda(), torch.FloatTensor(labels[val_lst]).cuda())
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)
    test_data=torch.utils.data.TensorDataset(torch.FloatTensor(data[test_lst]).cuda(), torch.FloatTensor(labels[test_lst]).cuda())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    print('==> Building model..')
    net = Net(n_feature=data.shape[-1], n_hidden=args.hidden, n_output=60, layers=args.layers, act=args.act).cuda() # 16 # 128 seems to be good
    net.apply(init_weights)
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss() 
    elif args.loss == 'sl1':
        criterion = nn.SmoothL1Loss() 
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-7)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.995, weight_decay=1e-7) #5e-4) # 0.99 0
    else:
        assert(0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    best_val_pcorr = np.zeros(60) 
    best_test_pcorr = np.zeros(60)
    fp = open(log_dir + '/' + args.model_name + '.txt', 'w')
    for epoch in range(args.epoch):
        # traniner
        tr_loss, tr_pcorr, data_time, batch_time = train(net, tr_loader, optimizer, criterion)
        scheduler.step() # shrink lr as 1/10 every args.step epochs
        _, val_pcorr = validation(net, val_loader, criterion)
        _, test_pcorr = validation(net, test_loader, criterion)
        print('Seed:{}, ep:{}, tr_loss:{:.4f}, tr_pcorr:{:.4f}, val_pcorr:{:.4f}, test_pcorr:{:.4f}'.format(
               args.seed, 
               epoch, tr_loss, tr_pcorr[0], val_pcorr[0], test_pcorr[0]))

        #if np.sum(val_pcorr) > np.sum(best_val_pcorr):
        #        best_val_pcorr = val_pcorr
        #        best_test_pcorr = test_pcorr
        #        torch.save(net.state_dict(), os.path.join(model_dir,'ckpt_best.pth'))

        temp = test_pcorr.tolist()
        temp = [str(t) for t in temp]
        fp.write(' '.join(temp) + '\n')

        # logging validation result. Currently no use as we use fixed epoch
        #temp2 = val_pcorr.tolist()
        #temp2 = [str(t) for t in temp2]
        #fp.write(' '.join(temp)+ '\t' + ' '.join(temp2) + '\n')
        if epoch > 0 and (epoch+1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(model_dir,'ckpt_{}.pth'.format(epoch+1)))
    fp.close()
        

def train(model, loader, optimizer, criterion):
    model.train()
    loss_agg = []
    ys_ = []
    ys = []
    data_time = AverageMeter()
    batch_time = AverageMeter()
    start = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        # log time
        data_time.update(time.time() - start)
        # zero-grad optimizer
        optimizer.zero_grad()
        # forward pass
        y_ = model(x.cuda())
        loss = criterion(y.cuda(), y_)
        # backward pass
        loss.backward()
        optimizer.step()
        # log loss and prediction
        loss_agg.append(loss.item())
        ys_.extend(y_.cpu().tolist())
        ys.extend(y.cpu().tolist())
        batch_time.update(time.time() - start)
        start = time.time()
    # calculate pcorr
    ys = np.array(ys).squeeze()
    ys_ = np.array(ys_).squeeze()
    corrs = np.zeros(60)
    for i in range(60):
        corrs[i], _ = pearsonr(ys[:,i], ys_[:,i])
    return sum(loss_agg)/len(loss_agg), corrs, data_time.avg, batch_time.avg

def validation(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        loss_agg = []
        ys_ = []
        ys = []
        for batch_idx, (x, y) in enumerate(loader):
            y_ = model(x.cuda())
            loss = criterion(y.cuda(), y_)
            loss_agg.append(loss.item())
            ys_.extend(y_.cpu().tolist())
            ys.extend(y.cpu().tolist())
            # saliency
    ys = np.array(ys).squeeze()
    ys_ = np.array(ys_).squeeze()
    corrs = np.zeros(60)
    for i in range(60):
        corrs[i], _ = pearsonr(ys[:,i], ys_[:,i])
    return sum(loss_agg)/len(loss_agg), corrs


# infernece and get the salincy map with gradient
def inference(model, loader, criterion):
    model.eval()
    loss_agg = []
    ys_ = []
    ys = []
    for batch_idx, (x, y) in enumerate(loader):
        y_ = model(x.cuda())
        loss = criterion(y.cuda(), y_)
        loss_agg.append(loss.item())
        ys_.extend(y_.cpu().tolist())
        ys.extend(y.cpu().tolist())
        # saliency
        sal=saliency(x.cuda(), y.cuda(), model, criterion)
        print(sal.shape)
    ys = np.array(ys).squeeze()
    ys_ = np.array(ys_).squeeze()
    corr, _ = pearsonr(ys, ys_)
    return sum(loss_agg)/len(loss_agg), corr

def saliency(X, y, model, criterion):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Wrap the input tensors in Variables
    #X_var = Variable(X, requires_grad=True)
    X.requires_grad=True
    saliency = None
    # Forward pass.
    y_ = model(X)
    loss = criterion(y, y_)
    # Get the correct class computed scores.
    #ccc = torch.ones_like(y_var).float()
    #ccc_var = Variable(ccc, requires_grad=False)
    # Backward pass, need to supply initial gradients of same tensor shape as scores.
    loss.backward()
    # Get gradient for image.
    saliency = X.grad.data
    # Convert from 3d to 1d.
    saliency = saliency.abs()
    print(saliency.size())
    saliency, i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze()
    return saliency



if __name__ == '__main__':
    main(args)

import torch
from torch.utils.data.dataset import Dataset
import os, argparse, json, time
from torch.autograd import Variable
import numpy as np
from utils import *
from model import *
from tensorboardX import SummaryWriter 
from scipy.stats import pearsonr
torch.manual_seed(1) 

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--kfold', default=10, type=int, help='kfold')
    parser.add_argument('--lr', default=0.000005, type=float, help='learning rate') # 0.000001
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--epoch', default=50, type=int, help='batch size')
    parser.add_argument('--loss', default='mse', type=str, help='loss')
    parser.add_argument('--leaky', action='store_true', help='use leaky relu')
    parser.add_argument('--layers', default=2, type=int, help='layers')
    parser.add_argument('--step', default=40, type=int, help='layers')
    parser.add_argument('--model_name', default='l1_hh128_lkrelu_b16_lr5e6', type=str, help='model_name')
    parser.add_argument('--hidden', default=128, type=int, help='hidden size')
    args = parser.parse_args()

    writer = SummaryWriter('runs/'+args.model_name)
    model_dir = os.path.join('models/', args.model_name)
    log_dir = os.path.join('log', args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # test_data
    data = np.load('data/subjects.npy')
    labels = 2.5*(np.load('data/neofac.npy')-0.3) # normalize to [0,1]
    splits = get_kfold(np.arange(data.shape[0]), k=args.kfold)

    for q in range(60):
        for k in range(args.kfold):
            train_lst, test_lst = splits[k]
            idx1, idx2 = get_kfold(train_lst, k=args.kfold)[0]
            validation_lst = train_lst[idx2]
            train_lst = train_lst[idx1]
            tr_data=torch.utils.data.TensorDataset(torch.FloatTensor(data[train_lst]).cuda(), torch.FloatTensor(np.expand_dims(labels[train_lst, q], axis=1)).cuda())
            tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=False)

            vali_data = torch.utils.data.TensorDataset(torch.FloatTensor(data[validation_lst]).cuda(), torch.FloatTensor(np.expand_dims(labels[validation_lst, q], axis=1)).cuda())
            vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

            test_data=torch.utils.data.TensorDataset(torch.FloatTensor(data[test_lst]).cuda(), torch.FloatTensor(np.expand_dims(labels[test_lst, q], axis=1)).cuda())
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

            # Model
            print('==> Building model..')
            net = Net(n_feature=data.shape[-1], n_hidden=args.hidden, n_output=1, layers=args.layers, leaky=args.leaky).cuda() # 16 # 128 seems to be good
            net.apply(init_weights)
            if args.loss == 'mse':
                criterion = nn.MSELoss() #SmoothL1Loss() # L1Loss()
            elif args.loss == 'l1':
                criterion = nn.L1Loss() #SmoothL1Loss() # L1Loss()
            elif args.loss == 'sl1':
                criterion = nn.SmoothL1Loss() # L1Loss()
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.995, weight_decay=5e-4) #5e-4) # 0.99 0
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

            best_pcorr = 0.  # best test accuracy
            for epoch in range(args.epoch):
                # traniner
                tr_loss, tr_pcorr, data_time, batch_time = train(net, tr_loader, optimizer, criterion)
                scheduler.step() # shrink lr as 1/10 every 40 epoch
                # validation/testing (report only the best val_pcorr as the test_pcorr)
                vali_loss, vali_pcorr = validation(net, vali_loader, criterion)
                test_loss, test_pcorr = validation(net, test_loader, criterion)
                # log with tensorboard
                writer.add_scalar('vali_loss {}-{}'.format(q, k), vali_loss, epoch)
                writer.add_scalar('vali_pcorr {}-{}'.format(q, k), vali_pcorr, epoch)
                writer.add_scalar('train_loss {}-{}'.format(q, k), tr_loss, epoch)
                writer.add_scalar('train_pcorr {}-{}'.format(q,k), tr_pcorr, epoch)
                print('Qid-fold:{}/{}, Epoch {}, tr_loss:{:.4f}, tr_pcorr:{:.4f}, val_loss:{:.4f}, val_pcorr:{:.4f}, best_pcorr:{:.4f}, test_pcorr:{:.4f}, data_time:{:.4f}, batch_time:{:.4f}'.format(
                       q, k, 
                       epoch, tr_loss, tr_pcorr, vali_loss, vali_pcorr, best_pcorr, test_pcorr,
                       data_time, batch_time))
                if vali_pcorr > best_pcorr:
                    best_pcorr = vali_pcorr
                    torch.save(net.state_dict(), os.path.join(model_dir,'ckpt_best.t7'))
            with open(log_dir + '/' + args.model_name + '_' + str(q) + '_' + str(k)+'.txt', 'w') as fp:
                fp.write(str(best_pcorr)+'\n')
        

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
    corr, _ = pearsonr(ys, ys_)
    return sum(loss_agg)/len(loss_agg), corr, data_time.avg, batch_time.avg

def validation(model, loader, criterion):
    model.eval()
    #with torch.no_grad():
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
    corr, _ = pearsonr(ys, ys_)
    return sum(loss_agg)/len(loss_agg), corr 


if __name__ == '__main__':
    main()

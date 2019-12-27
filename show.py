import numpy as np
import glob
import os, sys

log_dir = sys.argv[1]
max_ep=90
num_log = 0
logs = []
for m in glob.glob('{}/*/*.txt'.format(log_dir)):
    logs.append(m.strip())
    num_log += 1


ps = np.zeros((num_log, max_ep, 60))
for n, log in enumerate(logs):
    with open(log) as lines:
        for ep, line in enumerate(lines):
            temp = np.array(line.strip().split(' '))
            ps[n, ep, :] = temp

# select only the last(-1) ep
psm = ps[:,-1,:]
# show statistics
avg=np.mean(psm,axis=0)
std=np.std(psm,axis=0)
mm = np.max(psm,axis=0)
nn = np.min(psm,axis=0)
avg=np.mean(avg)
std=np.mean(std)
mm = np.mean(mm)
nn = np.mean(nn)
#print('ID|Mean|Std|Max|Min')
print("All:{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}".format(ep, avg, std, mm, nn))


# best (over 0~90 ep) stat
ps_max = np.max(ps, axis=1)
avg=np.mean(ps_max,axis=0)
std=np.std(ps_max,axis=0)
mm = np.max(ps_max,axis=0)
nn = np.min(ps_max,axis=0)
avg=np.mean(avg)
std=np.mean(std)
mm = np.mean(mm)
nn = np.mean(nn)
print("Best|{:.4f}|{:.4f}|{:.4f}|{:.4f}".format(avg, std, mm, nn))

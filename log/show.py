import numpy as np
import glob
import os, sys

#models = []
#with open(sys.argv[1]) as lines:
#    for line in lines:
#        if line.startswith('python'):
#            line = line.strip().split(' ')
#            idx = line.index('--model_name') + 1
#            models.append(line[idx])
model = sys.argv[1]
print(model)
print('ID|Mean|Std|Max|Min')
for i in range(60):
    ps = [] 
    for m in glob.glob('{}/{}_{}_*.txt'.format(model, model,i)):
        ps.append(float(open(m).read().strip()))
    avg=np.mean(ps)
    std=np.std(ps)
    mm = np.max(ps)
    nn = np.min(ps)
    print("{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}".format(i, avg, std, mm, nn))
print(model)


# HCP
Local connectome fingerprints of HCP 1062 subjects for prediction  

# USAGE
usage: main.py [-h] [--kfold KFOLD] [--lr LR] [--bs BS] [--epoch EPOCH]  
               [--loss LOSS] [--leaky] [--layers LAYERS]  
               [--model_name MODEL_NAME] [--hidden HIDDEN]  
# Example
python main.py --model_name l1_hh128_relu_b2_lr2e6 --loss l1 --lr 0.000002 --layers 3 --bs 2   
Check go.sh for an example script  

# Visualization/Analysis  
tensorboard --logdir=run/  
python log/show.py [your script_file]  

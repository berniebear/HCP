# Local connectome fingerprints of HCP 1062 subjects for prediction. 

We provide a simple model for training/testing the NEO-FFI Prediction Task using PyTorch. A 10-fold model for one question takes 2.5min with batchsize 16 with a 1080 Ti GPU. Cross-validation time For 60 quesitons is around 2.5 hours. Data is available at [here](http://dsi-studio.labsolver.org/download-images/local-connectome-fingerprints-of-hcp-1062-subjects-for-neofac-prediction). Please contact [Feng-Cheng Yeh](mailto:frank.yeh@gmail.com) or [Po-Yao Huang](mailto:poyaoh@cs.cmu.edu) if you have any questions.

# Usage
    python main.py [-h] [--kfold KFOLD] [--lr LR] [--bs BS] [--epoch EPOCH] [--loss LOSS] [--leaky] [--layers LAYERS] [--model_name MODEL_NAME] [--hidden HIDDEN]

# Example
For a 3-layer MLP model with batch size 16 and 2e-6 learning rate and l1 loss, the command is:

    python main.py --model_name l1_hh128_relu_b2_lr2e6 --loss l1 --lr 0.000002 --layers 3 --bs 16
Please check go.sh for an example script which trains 3 models sequentially.

# Visualization
    tensorboard --logdir=run/
    
# Analysis
The performance log file is stored at ./logs/[model_name]. We provide a script to show the reslts:

    python log/show.py [model_name]
    
# Performance
Model Name:l1_hh128_relu_b16_lr1e5
Metric: Pearson Corr


|ID|Mean|STD|Max|Min|
|---|---|---|---|---|
0|0.2906|0.0626|0.4469|0.2282
1|0.1591|0.0846|0.2881|0.0214
2|0.1790|0.0744|0.3188|0.0295
3|0.0750|0.0559|0.1791|0.0000
4|0.1548|0.0831|0.2593|0.0000
5|0.1568|0.0372|0.2049|0.0896
6|0.1383|0.0893|0.3318|0.0117
7|0.1432|0.0783|0.2486|0.0212
8|0.2262|0.0705|0.3226|0.1234
9|0.1358|0.0660|0.2427|0.0000
10|0.2062|0.0742|0.3014|0.0482
11|0.1062|0.0751|0.2440|0.0000
12|0.0677|0.0466|0.1535|0.0000
13|0.1849|0.0948|0.3260|0.0490
14|0.1304|0.0407|0.1932|0.0471
15|0.1456|0.0845|0.3223|0.0066
16|0.1327|0.0580|0.2504|0.0379
17|0.1656|0.0587|0.2828|0.0732
18|0.1229|0.0682|0.2388|0.0203
19|0.1041|0.0845|0.2745|0.0000
20|0.1248|0.0446|0.2139|0.0804
21|0.2079|0.0677|0.3425|0.1255
22|0.1171|0.0753|0.2064|0.0034
23|0.1643|0.0777|0.3261|0.0284
24|0.1483|0.0606|0.3129|0.0966
25|0.1424|0.0906|0.3063|0.0000
26|0.1939|0.0751|0.3668|0.1004
27|0.1513|0.0864|0.3198|0.0308
28|0.2959|0.0658|0.4176|0.2155
29|0.2503|0.0398|0.3185|0.1713
30|0.1294|0.0882|0.2591|0.0000
31|0.1811|0.0865|0.2860|0.0254
32|0.1950|0.0834|0.3651|0.0902
33|0.1136|0.0711|0.2347|0.0025
34|0.0925|0.0460|0.1727|0.0182
35|0.2372|0.0794|0.3468|0.1087
36|0.1140|0.0448|0.2065|0.0250
37|0.2456|0.0918|0.3746|0.1146
38|0.2886|0.0411|0.3478|0.2254
39|0.1038|0.0865|0.2118|0.0000
40|0.1596|0.0384|0.2228|0.0955
41|0.1282|0.0576|0.2475|0.0526
42|0.1474|0.0452|0.2208|0.0788
43|0.1412|0.0717|0.2602|0.0645
44|0.2042|0.0761|0.3545|0.1036
45|0.2363|0.0808|0.4446|0.1395
46|0.1439|0.0768|0.2507|0.0141
47|0.2336|0.0863|0.3848|0.0627
48|0.0841|0.0735|0.2026|0.0000
49|0.1299|0.0664|0.2383|0.0000
50|0.1447|0.0648|0.2452|0.0156
51|0.1558|0.0733|0.2776|0.0325
52|0.2446|0.0992|0.4472|0.1389
53|0.2245|0.0440|0.3206|0.1715
54|0.1446|0.0515|0.2523|0.0689
55|0.1231|0.0546|0.2013|0.0359
56|0.1314|0.0642|0.2210|0.0148
57|0.2150|0.0495|0.2861|0.1302
58|0.1645|0.0561|0.2571|0.0961
59|0.1059|0.0829|0.2619|0.0000

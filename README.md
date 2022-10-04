# Foreknow
We propose a Graph Attention Network (GAT)-based model for predicting credit default risk, leveraging various types of data.

# Requirements
pytorch
pytorch-geometric

# Usage
## 1. Download dataset
https://pan.baidu.com/s/1q1spkljIAeaogkFRBz3XfQ?pwd=cpbu
## 2. Process data
2.1 ```process.py```：We select some relevant features according to previous studies (Zhang et al., 2020; Xia et al., 2021; Liu et al., 2022) and expert knowledge. Select the maximum or mean value of some attributes, extra numerical attributes to represent to represent each categorical attribute.

input:"application_train.csv"、"bureau.csv"、"credit_card_balance.csv"

output:"one.csv"、"binary.csv"、"r1_onehot.csv"、"d1_onehot.csv"、"l1_onehot.csv"、"r1.csv"、"l1.csv"

2.2 ```A&D_distance.Rmd```:Calculation of mixed data distances by Ahmad & Dey method

input:"r1.csv"、"l1.csv"

output:"ahmad_r1.csv"、"ahmadl1.csv"

## 3. model
3.1 ```myGAT.py```:our method. We conduct various experiments to verify the effectiveness of our proposed GAT-based model, including comparison with baseline methods, with extended baseline methods, component analysis, feature significance analysis and impact of parameters.

input:"r1_onehot.csv"、"d1_onehot.csv"、"l1_onehot.csv"、"ahmad_r1.csv"、"ahmadl1.csv"、"binary.csv"

output:"pre.csv"

# Foreknow
Forecasting Credit Default Risk with Graph Attention Networks

# Requirements
pytorch
pytorch-geometric

# Usage
## ```data_process```
1."relate_screen.ipynb"：Associating three tables, filtering variables
input:"application_train.csv"、"bureau.csv"、"credit_card_balance.csv"
output:"filtering.csv"

2."merge.ipynb"：Processed as one piece of data per person
input:"filtering.csv"
output:"one.csv"、"binary.csv"、"r1_onehot.csv"、"d1_onehot.csv"、"l1_onehot.csv"、"r1.csv"、"l1.csv"

3."A&D_distance.Rmd":Calculation of mixed data distances by Ahmad & Dey method
input:"r1.csv"、"l1.csv"
output:"ahmad_r1.csv"、"ahmadl1.csv"

## ```model```
1."myGAT.ipynb":our method
input:"r1_onehot.csv"、"d1_onehot.csv"、"l1_onehot.csv"、"ahmad_r1.csv"、"ahmadl1.csv"、"binary.csv"
output:"pre.csv"

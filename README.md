# Foreknow
This is a Pytorch implementation of the model in the paper "Forecasting Credit Default Risk with Graph Attention Networks". 

# Requirements
pytorch

pytorch-geometric

# Usage
## To run the code
1. Download the datasets at  [here](https://pan.baidu.com/s/1q1spkljIAeaogkFRBz3XfQ?pwd=cpbu). 
2. Run the data processing code: 

   ```python process.py```

3. Run the credit default risk prediction code: 

   ```python myGAT.py```

## Code Organization
1. ```process.py```: This file is used to process the three raw datasets and output relevant attributes for future credit default risk prediction tasks. 
* Input: "application_train.csv", "bureau.csv", "credit_card_balance.csv"
* Output: "one.csv"、"binary.csv", "r1_onehot.csv", "d1_onehot.csv", "l1_onehot.csv", "r1.csv", "l1.csv"

2. ```A&D_distance.Rmd```: This file is used to calculate distances between mixed data by Ahmad & Dey method.
* Input: "r1.csv", "l1.csv"
* Output: "ahmad_r1.csv", "ahmadl1.csv"

3. ```myGAT.py```: This file is the core implementation of the prediction model. 
* Input: "r1_onehot.csv", "d1_onehot.csv", "l1_onehot.csv", "ahmad_r1.csv", "ahmadl1.csv", "binary.csv"
* Output: "pre.csv"

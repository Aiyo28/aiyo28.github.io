---
layout: post
title:  "Credit Card Fraud Detection"
date:   2020-03-09 14:42:25
categories: mediator feature
tags: featured
image: /assets/article_images/2020-03-09-CCFD-solutions/CCFD.jpg
---

# Credit Card Fraud Detection

## Identify fraudulent credit card transactions.

- Defining the problem statement
- Collecting the data
- Exploratory data analysis
- Modelling
- Testing

### Download the data
You can find this dataset in the following link: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## 1. Defining the problem statement

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

### Content
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

### Columns  
`Time` - Number of seconds elapsed between this transaction and the first transaction in the dataset.  
`V1-V28` - may be result of a PCA Dimensionality reduction to protect user identities and sensitive features.  
`Amount` - Transaction amount.  
`Class` - 1 for fraudulent transactions, 0 otherwise.  

## 2. We will need to download all relevant dependencies we need.

as well as upload train and test datasets


```python
# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dependencies
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
```

### Loading in the data

After we have downloaded the data, we need to get it into the notebook. We will import our dataset from a .csv file as a Pandas DataFrame. Furthermore, we will begin exploring the dataset to gain an understanding of the type, quantity, and distribution of data in our dataset.


```python
# load the data using pandas
data = pd.read_csv(r'data/creditcard.csv')
```

## 3. Exploration of the data


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
# Start exploring the dataset
print(data.columns)
```

    Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')



```python
# Any null values?
data.isnull().sum().max()
```




    0




```python
# Print the shape of the data
data = data.sample(frac=0.1, random_state = 1)
print(data.shape)
print(data.describe())
```

    (28481, 31)
                    Time            V1            V2            V3            V4  \
    count   28481.000000  28481.000000  28481.000000  28481.000000  28481.000000   
    mean    94705.035216     -0.001143     -0.018290      0.000795      0.000350   
    std     47584.727034      1.994661      1.709050      1.522313      1.420003   
    min         0.000000    -40.470142    -63.344698    -31.813586     -5.266509   
    25%     53924.000000     -0.908809     -0.610322     -0.892884     -0.847370   
    50%     84551.000000      0.031139      0.051775      0.178943     -0.017692   
    75%    139392.000000      1.320048      0.792685      1.035197      0.737312   
    max    172784.000000      2.411499     17.418649      4.069865     16.715537   

                     V5            V6            V7            V8            V9  \
    count  28481.000000  28481.000000  28481.000000  28481.000000  28481.000000   
    mean      -0.015666      0.003634     -0.008523     -0.003040      0.014536   
    std        1.395552      1.334985      1.237249      1.204102      1.098006   
    min      -42.147898    -19.996349    -22.291962    -33.785407     -8.739670   
    25%       -0.703986     -0.765807     -0.562033     -0.208445     -0.632488   
    50%       -0.068037     -0.269071      0.028378      0.024696     -0.037100   
    75%        0.603574      0.398839      0.559428      0.326057      0.621093   
    max       28.762671     22.529298     36.677268     19.587773      8.141560   

           ...           V21           V22           V23           V24  \
    count  ...  28481.000000  28481.000000  28481.000000  28481.000000   
    mean   ...      0.004740      0.006719     -0.000494     -0.002626   
    std    ...      0.744743      0.728209      0.645945      0.603968   
    min    ...    -16.640785    -10.933144    -30.269720     -2.752263   
    25%    ...     -0.224842     -0.535877     -0.163047     -0.360582   
    50%    ...     -0.029075      0.014337     -0.012678      0.038383   
    75%    ...      0.189068      0.533936      0.148065      0.434851   
    max    ...     22.588989      6.090514     15.626067      3.944520   

                    V25           V26           V27           V28        Amount  \
    count  28481.000000  28481.000000  28481.000000  28481.000000  28481.000000   
    mean      -0.000917      0.004762     -0.001689     -0.004154     89.957884   
    std        0.520679      0.488171      0.418304      0.321646    270.894630   
    min       -7.025783     -2.534330     -8.260909     -9.617915      0.000000   
    25%       -0.319611     -0.328476     -0.071712     -0.053379      5.980000   
    50%        0.015231     -0.049750      0.000914      0.010753     22.350000   
    75%        0.351466      0.253580      0.090329      0.076267     78.930000   
    max        5.541598      3.118588     11.135740     15.373170  19656.530000   

                  Class  
    count  28481.000000  
    mean       0.001720  
    std        0.041443  
    min        0.000000  
    25%        0.000000  
    50%        0.000000  
    75%        0.000000  
    max        1.000000  

    [8 rows x 31 columns]



```python
# Plot histograms of each parameter
data.hist(figsize = (20, 20))
plt.show()
```


![png](/assets/article_images/2020-03-09-CCFD-solutions/CCFD_solutions_files/CCFD_solutions_8_0.png)



```python
# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
```

    0.0017234102419808666
    Fraud Cases: 49
    Valid Transactions: 28432



```python
# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()
```


![png](/assets/article_images/2020-03-09-CCFD-solutions/CCFD_solutions_files/CCFD_solutions_10_0.png)



```python
# Get all the columns from the dataFrame
columns = data.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)
```

    (28481, 30)
    (28481,)


## 4. Isolation of Outliers

Now we can begin deploying our machine learning algorithms. We will use the following techniques:

**Local Outlier Factor (LOF)**

The local outlier factor is based on a concept of a local density, where locality is given by
*k* nearest neighbors, whose distance is used to estimate the density. By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. These are considered to be outliers (Breunig, M. M.).

**Isolation Forest Algorithm**

Isolation forest is an unsupervised learning algorithm for anomaly detection that works on the principle of isolating anomalies, instead of the most common techniques of profiling normal points (Liu).

Since reacuring devisions can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.


```python
# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X), contamination=outlier_fraction, random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)
}

# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):

    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Reshape the prediction values to 0 for valid, 1 for fraud.
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
```

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/iforest.py:247: FutureWarning: behaviour="old" is deprecated and will be removed in version 0.22. Please use behaviour="new", which makes the decision_function change to match other anomaly detection algorithm API.
      FutureWarning)
    /opt/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/iforest.py:415: DeprecationWarning: threshold_ attribute is deprecated in 0.20 and will be removed in 0.22.
      " be removed in 0.22.", DeprecationWarning)


    Isolation Forest: 71
    0.99750711000316
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     28432
               1       0.28      0.29      0.28        49

        accuracy                           1.00     28481
       macro avg       0.64      0.64      0.64     28481
    weighted avg       1.00      1.00      1.00     28481

    Local Outlier Factor: 97
    0.9965942207085425
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     28432
               1       0.02      0.02      0.02        49

        accuracy                           1.00     28481
       macro avg       0.51      0.51      0.51     28481
    weighted avg       1.00      1.00      1.00     28481




    <Figure size 648x504 with 0 Axes>


### Reference

1.  Breunig, M. M.; Kriegel, H.-P.; Ng, R. T.; Sander, J. (2000). LOF: Identifying Density-based Local Outliers (PDF). Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. SIGMOD. pp. 93–104. doi:10.1145/335191.335388. ISBN 1-58113-217-4.

2. Liu, Fei Tony; Ting, Kai Ming; Zhou, Zhi-Hua (December 2008). "Isolation Forest". 2008 Eighth IEEE International Conference on Data Mining: 413–422. doi:10.1109/ICDM.2008.17. ISBN 978-0-7695-3502-9.

3. This work has been inspired and modeled after
[Credit Card Fraud Detection Eduonix Solution](https://www.kaggle.com/sundarshahi/credit-card-fraud-detection-eduonix-solution/notebook)

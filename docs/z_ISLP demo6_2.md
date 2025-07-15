```python
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.linear_model as skl
import sklearn.model_selection as skm

from ISLP import load_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from matplotlib.pyplot import subplots
```

```python
hits = load_data('Hitters')
hits.info()
hits.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 322 entries, 0 to 321
    Data columns (total 20 columns):
     #   Column     Non-Null Count  Dtype   
    ---  ------     --------------  -----   
     0   AtBat      322 non-null    int64   
     1   Hits       322 non-null    int64   
     2   HmRun      322 non-null    int64   
     3   Runs       322 non-null    int64   
     4   RBI        322 non-null    int64   
     5   Walks      322 non-null    int64   
     6   Years      322 non-null    int64   
     7   CAtBat     322 non-null    int64   
     8   CHits      322 non-null    int64   
     9   CHmRun     322 non-null    int64   
     10  CRuns      322 non-null    int64   
     11  CRBI       322 non-null    int64   
     12  CWalks     322 non-null    int64   
     13  League     322 non-null    category
     14  Division   322 non-null    category
     15  PutOuts    322 non-null    int64   
     16  Assists    322 non-null    int64   
     17  Errors     322 non-null    int64   
     18  Salary     263 non-null    float64 
     19  NewLeague  322 non-null    category
    dtypes: category(3), float64(1), int64(16)
    memory usage: 44.2 KB

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
      <th>AtBat</th>
      <th>Hits</th>
      <th>HmRun</th>
      <th>Runs</th>
      <th>RBI</th>
      <th>Walks</th>
      <th>Years</th>
      <th>CAtBat</th>
      <th>CHits</th>
      <th>CHmRun</th>
      <th>CRuns</th>
      <th>CRBI</th>
      <th>CWalks</th>
      <th>League</th>
      <th>Division</th>
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>Salary</th>
      <th>NewLeague</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>293</td>
      <td>66</td>
      <td>1</td>
      <td>30</td>
      <td>29</td>
      <td>14</td>
      <td>1</td>
      <td>293</td>
      <td>66</td>
      <td>1</td>
      <td>30</td>
      <td>29</td>
      <td>14</td>
      <td>A</td>
      <td>E</td>
      <td>446</td>
      <td>33</td>
      <td>20</td>
      <td>NaN</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>315</td>
      <td>81</td>
      <td>7</td>
      <td>24</td>
      <td>38</td>
      <td>39</td>
      <td>14</td>
      <td>3449</td>
      <td>835</td>
      <td>69</td>
      <td>321</td>
      <td>414</td>
      <td>375</td>
      <td>N</td>
      <td>W</td>
      <td>632</td>
      <td>43</td>
      <td>10</td>
      <td>475.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>479</td>
      <td>130</td>
      <td>18</td>
      <td>66</td>
      <td>72</td>
      <td>76</td>
      <td>3</td>
      <td>1624</td>
      <td>457</td>
      <td>63</td>
      <td>224</td>
      <td>266</td>
      <td>263</td>
      <td>A</td>
      <td>W</td>
      <td>880</td>
      <td>82</td>
      <td>14</td>
      <td>480.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>496</td>
      <td>141</td>
      <td>20</td>
      <td>65</td>
      <td>78</td>
      <td>37</td>
      <td>11</td>
      <td>5628</td>
      <td>1575</td>
      <td>225</td>
      <td>828</td>
      <td>838</td>
      <td>354</td>
      <td>N</td>
      <td>E</td>
      <td>200</td>
      <td>11</td>
      <td>3</td>
      <td>500.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>321</td>
      <td>87</td>
      <td>10</td>
      <td>39</td>
      <td>42</td>
      <td>30</td>
      <td>2</td>
      <td>396</td>
      <td>101</td>
      <td>12</td>
      <td>48</td>
      <td>46</td>
      <td>33</td>
      <td>N</td>
      <td>E</td>
      <td>805</td>
      <td>40</td>
      <td>4</td>
      <td>91.5</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>

```python
hits.describe()
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
      <th>AtBat</th>
      <th>Hits</th>
      <th>HmRun</th>
      <th>Runs</th>
      <th>RBI</th>
      <th>Walks</th>
      <th>Years</th>
      <th>CAtBat</th>
      <th>CHits</th>
      <th>CHmRun</th>
      <th>CRuns</th>
      <th>CRBI</th>
      <th>CWalks</th>
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.00000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>322.000000</td>
      <td>263.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>380.928571</td>
      <td>101.024845</td>
      <td>10.770186</td>
      <td>50.909938</td>
      <td>48.027950</td>
      <td>38.742236</td>
      <td>7.444099</td>
      <td>2648.68323</td>
      <td>717.571429</td>
      <td>69.490683</td>
      <td>358.795031</td>
      <td>330.118012</td>
      <td>260.239130</td>
      <td>288.937888</td>
      <td>106.913043</td>
      <td>8.040373</td>
      <td>535.925882</td>
    </tr>
    <tr>
      <th>std</th>
      <td>153.404981</td>
      <td>46.454741</td>
      <td>8.709037</td>
      <td>26.024095</td>
      <td>26.166895</td>
      <td>21.639327</td>
      <td>4.926087</td>
      <td>2324.20587</td>
      <td>654.472627</td>
      <td>86.266061</td>
      <td>334.105886</td>
      <td>333.219617</td>
      <td>267.058085</td>
      <td>280.704614</td>
      <td>136.854876</td>
      <td>6.368359</td>
      <td>451.118681</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>19.00000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>67.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>255.250000</td>
      <td>64.000000</td>
      <td>4.000000</td>
      <td>30.250000</td>
      <td>28.000000</td>
      <td>22.000000</td>
      <td>4.000000</td>
      <td>816.75000</td>
      <td>209.000000</td>
      <td>14.000000</td>
      <td>100.250000</td>
      <td>88.750000</td>
      <td>67.250000</td>
      <td>109.250000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>190.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>379.500000</td>
      <td>96.000000</td>
      <td>8.000000</td>
      <td>48.000000</td>
      <td>44.000000</td>
      <td>35.000000</td>
      <td>6.000000</td>
      <td>1928.00000</td>
      <td>508.000000</td>
      <td>37.500000</td>
      <td>247.000000</td>
      <td>220.500000</td>
      <td>170.500000</td>
      <td>212.000000</td>
      <td>39.500000</td>
      <td>6.000000</td>
      <td>425.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>512.000000</td>
      <td>137.000000</td>
      <td>16.000000</td>
      <td>69.000000</td>
      <td>64.750000</td>
      <td>53.000000</td>
      <td>11.000000</td>
      <td>3924.25000</td>
      <td>1059.250000</td>
      <td>90.000000</td>
      <td>526.250000</td>
      <td>426.250000</td>
      <td>339.250000</td>
      <td>325.000000</td>
      <td>166.000000</td>
      <td>11.000000</td>
      <td>750.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>687.000000</td>
      <td>238.000000</td>
      <td>40.000000</td>
      <td>130.000000</td>
      <td>121.000000</td>
      <td>105.000000</td>
      <td>24.000000</td>
      <td>14053.00000</td>
      <td>4256.000000</td>
      <td>548.000000</td>
      <td>2165.000000</td>
      <td>1659.000000</td>
      <td>1566.000000</td>
      <td>1378.000000</td>
      <td>492.000000</td>
      <td>32.000000</td>
      <td>2460.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
hits = hits.dropna()
sns.pairplot(hits, hue="Salary")
```

    <seaborn.axisgrid.PairGrid at 0x72846cfd77d0>
    
![png](z_ISLP%20demo6_2_files/z_ISLP%20demo6_2_3_1.png)

```python
for col in ["League", "Division", "NewLeague"]:
    hits[col] = [1 if val == hits[col].iloc[0] else 0 for val in hits[col]]

X = hits.drop(columns=["Salary"])
y = hits["Salary"]
lambdas = 10**np.linspace(8, -2, 100)
```

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.1, random_state=0)
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=0)
```

```python
model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('ridge', skl.RidgeCV(alphas=lambdas, store_cv_results=True))
    ]
)
```

```python
model.fit(X_train, y_train)
model_pred = model.predict(X_test)
MSE(model_pred, y_test), R2(model_pred, y_test)
```

    (99295.86827979995, 0.4448133917882484)

```python
ridgecv = model.named_steps['ridge']
err_path = ridgecv.cv_results_.mean(0)

fig, ax = subplots()
ax.plot(-np.log10(lambdas), err_path)
ax.axvline(-np.log10(ridgecv.alpha_), c='k', ls='--')
ax.set_ylim ([50000 ,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross -validated MSE', fontsize=20);
```

    <>:8: SyntaxWarning: invalid escape sequence '\l'
    <>:8: SyntaxWarning: invalid escape sequence '\l'
    /tmp/ipykernel_31051/3379189933.py:8: SyntaxWarning: invalid escape sequence '\l'
      ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
    
![png](z_ISLP%20demo6_2_files/z_ISLP%20demo6_2_8_1.png)

```python
ridgecv.coef_, ridgecv.alpha_
```

    (array([-2.43878762e+02,  2.60488628e+02, -1.03873887e-01, -1.94743085e+01,
            -2.47831048e+00,  1.31955900e+02, -4.79502021e+01, -9.65621490e+01,
             1.12083526e+02,  9.42043348e+01,  2.26196846e+02,  6.47739865e+01,
            -1.56211768e+02,  2.56043840e+01, -6.22373727e+01,  7.81558996e+01,
             4.86000393e+01, -2.90786464e+01, -7.98183554e+00]),
     2.6560877829466896)

```python
model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('lasso', skl.LassoCV(alphas=lambdas))
    ]
)
```

```python
model.fit(X_train, y_train)
model_pred = model.predict(X_test)
MSE(model_pred, y_test), R2(model_pred, y_test)
```

    (100781.91070990668, 0.43806280005181397)

```python
lassocv = model.named_steps['lasso']
err_path = lassocv.mse_path_.mean(1)

fig, ax = subplots()
ax.plot(-np.log10(lambdas), err_path)
ax.axvline(-np.log10(lassocv.alpha_), c='k', ls='--')
ax.set_ylim ([50000 ,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross -validated MSE', fontsize=20);
```

    <>:8: SyntaxWarning: invalid escape sequence '\l'
    <>:8: SyntaxWarning: invalid escape sequence '\l'
    /tmp/ipykernel_31051/1786576076.py:8: SyntaxWarning: invalid escape sequence '\l'
      ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
    
![png](z_ISLP%20demo6_2_files/z_ISLP%20demo6_2_12_1.png)

```python
lassocv.coef_, lassocv.alpha_
```

    (array([-285.68139176,  298.36986021,   -0.        ,  -16.64102469,
               0.        ,  132.59353528,  -39.83682438,   -0.        ,
               0.        ,  100.62310668,  270.53870113,   33.4568784 ,
            -168.6634666 ,   17.83798661,  -59.0206921 ,   78.97330083,
              41.10553871,  -19.38100744,    0.        ]),
     2.104904144512022)

```python

```

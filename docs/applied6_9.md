```python
import numpy as np
import pandas as pd

from matplotlib.pyplot import subplots
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import seaborn as sns
from sklearn.metrics import r2_score as R2, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from ISLP import load_data
```


```python
college = load_data('College')
college.info()
college
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 777 entries, 0 to 776
    Data columns (total 18 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   Private      777 non-null    category
     1   Apps         777 non-null    int64   
     2   Accept       777 non-null    int64   
     3   Enroll       777 non-null    int64   
     4   Top10perc    777 non-null    int64   
     5   Top25perc    777 non-null    int64   
     6   F.Undergrad  777 non-null    int64   
     7   P.Undergrad  777 non-null    int64   
     8   Outstate     777 non-null    int64   
     9   Room.Board   777 non-null    int64   
     10  Books        777 non-null    int64   
     11  Personal     777 non-null    int64   
     12  PhD          777 non-null    int64   
     13  Terminal     777 non-null    int64   
     14  S.F.Ratio    777 non-null    float64 
     15  perc.alumni  777 non-null    int64   
     16  Expend       777 non-null    int64   
     17  Grad.Rate    777 non-null    int64   
    dtypes: category(1), float64(1), int64(16)
    memory usage: 104.2 KB





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
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
      <td>1660</td>
      <td>1232</td>
      <td>721</td>
      <td>23</td>
      <td>52</td>
      <td>2885</td>
      <td>537</td>
      <td>7440</td>
      <td>3300</td>
      <td>450</td>
      <td>2200</td>
      <td>70</td>
      <td>78</td>
      <td>18.1</td>
      <td>12</td>
      <td>7041</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>2186</td>
      <td>1924</td>
      <td>512</td>
      <td>16</td>
      <td>29</td>
      <td>2683</td>
      <td>1227</td>
      <td>12280</td>
      <td>6450</td>
      <td>750</td>
      <td>1500</td>
      <td>29</td>
      <td>30</td>
      <td>12.2</td>
      <td>16</td>
      <td>10527</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
      <td>1428</td>
      <td>1097</td>
      <td>336</td>
      <td>22</td>
      <td>50</td>
      <td>1036</td>
      <td>99</td>
      <td>11250</td>
      <td>3750</td>
      <td>400</td>
      <td>1165</td>
      <td>53</td>
      <td>66</td>
      <td>12.9</td>
      <td>30</td>
      <td>8735</td>
      <td>54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yes</td>
      <td>417</td>
      <td>349</td>
      <td>137</td>
      <td>60</td>
      <td>89</td>
      <td>510</td>
      <td>63</td>
      <td>12960</td>
      <td>5450</td>
      <td>450</td>
      <td>875</td>
      <td>92</td>
      <td>97</td>
      <td>7.7</td>
      <td>37</td>
      <td>19016</td>
      <td>59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yes</td>
      <td>193</td>
      <td>146</td>
      <td>55</td>
      <td>16</td>
      <td>44</td>
      <td>249</td>
      <td>869</td>
      <td>7560</td>
      <td>4120</td>
      <td>800</td>
      <td>1500</td>
      <td>76</td>
      <td>72</td>
      <td>11.9</td>
      <td>2</td>
      <td>10922</td>
      <td>15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>772</th>
      <td>No</td>
      <td>2197</td>
      <td>1515</td>
      <td>543</td>
      <td>4</td>
      <td>26</td>
      <td>3089</td>
      <td>2029</td>
      <td>6797</td>
      <td>3900</td>
      <td>500</td>
      <td>1200</td>
      <td>60</td>
      <td>60</td>
      <td>21.0</td>
      <td>14</td>
      <td>4469</td>
      <td>40</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Yes</td>
      <td>1959</td>
      <td>1805</td>
      <td>695</td>
      <td>24</td>
      <td>47</td>
      <td>2849</td>
      <td>1107</td>
      <td>11520</td>
      <td>4960</td>
      <td>600</td>
      <td>1250</td>
      <td>73</td>
      <td>75</td>
      <td>13.3</td>
      <td>31</td>
      <td>9189</td>
      <td>83</td>
    </tr>
    <tr>
      <th>774</th>
      <td>Yes</td>
      <td>2097</td>
      <td>1915</td>
      <td>695</td>
      <td>34</td>
      <td>61</td>
      <td>2793</td>
      <td>166</td>
      <td>6900</td>
      <td>4200</td>
      <td>617</td>
      <td>781</td>
      <td>67</td>
      <td>75</td>
      <td>14.4</td>
      <td>20</td>
      <td>8323</td>
      <td>49</td>
    </tr>
    <tr>
      <th>775</th>
      <td>Yes</td>
      <td>10705</td>
      <td>2453</td>
      <td>1317</td>
      <td>95</td>
      <td>99</td>
      <td>5217</td>
      <td>83</td>
      <td>19840</td>
      <td>6510</td>
      <td>630</td>
      <td>2115</td>
      <td>96</td>
      <td>96</td>
      <td>5.8</td>
      <td>49</td>
      <td>40386</td>
      <td>99</td>
    </tr>
    <tr>
      <th>776</th>
      <td>Yes</td>
      <td>2989</td>
      <td>1855</td>
      <td>691</td>
      <td>28</td>
      <td>63</td>
      <td>2988</td>
      <td>1726</td>
      <td>4990</td>
      <td>3560</td>
      <td>500</td>
      <td>1250</td>
      <td>75</td>
      <td>75</td>
      <td>18.1</td>
      <td>28</td>
      <td>4509</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
<p>777 rows × 18 columns</p>
</div>




```python
college.describe()
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
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3001.638353</td>
      <td>2018.804376</td>
      <td>779.972973</td>
      <td>27.558559</td>
      <td>55.796654</td>
      <td>3699.907336</td>
      <td>855.298584</td>
      <td>10440.669241</td>
      <td>4357.526384</td>
      <td>549.380952</td>
      <td>1340.642214</td>
      <td>72.660232</td>
      <td>79.702703</td>
      <td>14.089704</td>
      <td>22.743887</td>
      <td>9660.171171</td>
      <td>65.46332</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3870.201484</td>
      <td>2451.113971</td>
      <td>929.176190</td>
      <td>17.640364</td>
      <td>19.804778</td>
      <td>4850.420531</td>
      <td>1522.431887</td>
      <td>4023.016484</td>
      <td>1096.696416</td>
      <td>165.105360</td>
      <td>677.071454</td>
      <td>16.328155</td>
      <td>14.722359</td>
      <td>3.958349</td>
      <td>12.391801</td>
      <td>5221.768440</td>
      <td>17.17771</td>
    </tr>
    <tr>
      <th>min</th>
      <td>81.000000</td>
      <td>72.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>139.000000</td>
      <td>1.000000</td>
      <td>2340.000000</td>
      <td>1780.000000</td>
      <td>96.000000</td>
      <td>250.000000</td>
      <td>8.000000</td>
      <td>24.000000</td>
      <td>2.500000</td>
      <td>0.000000</td>
      <td>3186.000000</td>
      <td>10.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>776.000000</td>
      <td>604.000000</td>
      <td>242.000000</td>
      <td>15.000000</td>
      <td>41.000000</td>
      <td>992.000000</td>
      <td>95.000000</td>
      <td>7320.000000</td>
      <td>3597.000000</td>
      <td>470.000000</td>
      <td>850.000000</td>
      <td>62.000000</td>
      <td>71.000000</td>
      <td>11.500000</td>
      <td>13.000000</td>
      <td>6751.000000</td>
      <td>53.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1558.000000</td>
      <td>1110.000000</td>
      <td>434.000000</td>
      <td>23.000000</td>
      <td>54.000000</td>
      <td>1707.000000</td>
      <td>353.000000</td>
      <td>9990.000000</td>
      <td>4200.000000</td>
      <td>500.000000</td>
      <td>1200.000000</td>
      <td>75.000000</td>
      <td>82.000000</td>
      <td>13.600000</td>
      <td>21.000000</td>
      <td>8377.000000</td>
      <td>65.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3624.000000</td>
      <td>2424.000000</td>
      <td>902.000000</td>
      <td>35.000000</td>
      <td>69.000000</td>
      <td>4005.000000</td>
      <td>967.000000</td>
      <td>12925.000000</td>
      <td>5050.000000</td>
      <td>600.000000</td>
      <td>1700.000000</td>
      <td>85.000000</td>
      <td>92.000000</td>
      <td>16.500000</td>
      <td>31.000000</td>
      <td>10830.000000</td>
      <td>78.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48094.000000</td>
      <td>26330.000000</td>
      <td>6392.000000</td>
      <td>96.000000</td>
      <td>100.000000</td>
      <td>31643.000000</td>
      <td>21836.000000</td>
      <td>21700.000000</td>
      <td>8124.000000</td>
      <td>2340.000000</td>
      <td>6800.000000</td>
      <td>103.000000</td>
      <td>100.000000</td>
      <td>39.800000</td>
      <td>64.000000</td>
      <td>56233.000000</td>
      <td>118.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
college["Private"] = [1 if p == "Yes" else 0 for p in college["Private"]]
```


```python
sns.heatmap(college.corr())
```




    <Axes: >




    
![png](applied6_9_files/applied6_9_4_1.png)
    



```python
X = college.drop(columns=['Apps'])
y = college['Apps']

X_train, X_test, y_train, y_test = skm.train_test_split(X, y, shuffle=True, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
lambdas = 10**np.linspace(8, -4)
```


```python
lr = skl.LinearRegression()

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
```


```python
ridge = skl.RidgeCV(alphas=lambdas)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
```


```python
lasso = skl.LassoCV(alphas=lambdas)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
```


```python
pcr = Pipeline([
    ('pca', PCA()),
    ('plr', skl.LinearRegression())
])
param = {'pca__n_components': range(1, X_train.shape[1])}
pcr = skm.GridSearchCV(pcr, param_grid=param, scoring='r2')
pcr.fit(X_train, y_train)
pcr_pred = pcr.predict(X_test)
```


```python
pls = PLSRegression()
param = {'n_components': range(1, X_train.shape[1])}
pls = skm.GridSearchCV(pls, param_grid=param, scoring='r2')
pls.fit(X_train, y_train)
pls_pred = pls.predict(X_test)
```


```python
R2(y_test, lr_pred), R2(y_test, ridge_pred), R2(y_test, ridge_pred), R2(y_test, ridge_pred), R2(y_test, ridge_pred), 
```




    (0.9002392990734523,
     0.9002393892199767,
     0.9002393892199767,
     0.9002393892199767,
     0.9002393892199767)




```python
MSE(lr_pred, y_test), MSE(ridge_pred, y_test), MSE(lasso_pred, y_test), MSE(pcr_pred, y_test), MSE(pls_pred, y_test), 
```




    (1022430.0889255423,
     1022429.1650294779,
     1045905.0958773881,
     1025469.2195196956,
     1017373.0630376363)




```python
lr.coef_, ridge.coef_, lasso.coef_, pcr.best_estimator_.named_steps['plr'].coef_, pls.best_estimator_.coef_
```




    (array([-2.51082107e+02,  4.18269191e+03, -9.96433619e+02,  1.01537623e+03,
            -3.44032537e+02,  3.64813902e+02,  8.83658018e+01, -3.03380710e+02,
             1.91334277e+02,  2.53797453e+00, -8.05468486e+00, -1.82788586e+02,
            -4.51727150e+01,  1.50670386e+01,  8.14678768e+00,  2.49568955e+02,
             1.43324300e+02]),
     array([-2.51082354e+02,  4.18268552e+03, -9.96424071e+02,  1.01537358e+03,
            -3.44030511e+02,  3.64810149e+02,  8.83657476e+01, -3.03379720e+02,
             1.91334635e+02,  2.53800864e+00, -8.05477455e+00, -1.82788258e+02,
            -4.51729689e+01,  1.50671183e+01,  8.14624438e+00,  2.49569258e+02,
             1.43324391e+02]),
     array([-177.14499451, 3615.46802497,   -0.        ,  547.81793016,
              -0.        ,   -0.        ,    0.        ,  -54.17217488,
             120.98445659,    0.        ,   -0.        ,  -75.25305854,
             -32.24841232,   -0.        ,   -0.        ,  178.90491257,
              21.07618984]),
     array([ 384.41596737, 1610.37407754, -174.73373112, -582.86012121,
            1245.63502865, -335.64403864, -390.42036453, -371.47592549,
            -392.95424603, -193.05636542, -136.11556742,    4.19431906,
             125.97688297,  176.81414375, 2402.96762785, 2477.93764111]),
     array([[-2.50590747e+02,  4.13036908e+03, -6.75857579e+02,
              1.10640670e+03, -4.05532175e+02,  7.73330234e+01,
              1.08632225e+02, -3.31449999e+02,  1.94782720e+02,
             -1.10102229e+01,  1.66980506e+00, -2.19589615e+02,
             -1.05852444e+01,  2.03584653e+01,  2.05234569e+01,
              2.45313541e+02,  1.28515658e+02]]))




```python
ridge.alpha_, lasso.alpha_, pcr.best_params_, pls.best_params_
```




    (0.0001, 42.91934260128778, {'pca__n_components': 16}, {'n_components': 10})




```python

```


```python

```


```python

```

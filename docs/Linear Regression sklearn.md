```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

```python
from statsmodels.stats.outliers_influence \
     import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
```

```python
df = pd.read_csv("../dataset/Auto.csv", na_values='?')
df = df.dropna()
df = df.iloc[20:,:]
df
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>25.0</td>
      <td>4</td>
      <td>110.0</td>
      <td>87.0</td>
      <td>2672</td>
      <td>17.5</td>
      <td>70</td>
      <td>2</td>
      <td>peugeot 504</td>
    </tr>
    <tr>
      <th>21</th>
      <td>24.0</td>
      <td>4</td>
      <td>107.0</td>
      <td>90.0</td>
      <td>2430</td>
      <td>14.5</td>
      <td>70</td>
      <td>2</td>
      <td>audi 100 ls</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25.0</td>
      <td>4</td>
      <td>104.0</td>
      <td>95.0</td>
      <td>2375</td>
      <td>17.5</td>
      <td>70</td>
      <td>2</td>
      <td>saab 99e</td>
    </tr>
    <tr>
      <th>23</th>
      <td>26.0</td>
      <td>4</td>
      <td>121.0</td>
      <td>113.0</td>
      <td>2234</td>
      <td>12.5</td>
      <td>70</td>
      <td>2</td>
      <td>bmw 2002</td>
    </tr>
    <tr>
      <th>24</th>
      <td>21.0</td>
      <td>6</td>
      <td>199.0</td>
      <td>90.0</td>
      <td>2648</td>
      <td>15.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc gremlin</td>
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
    </tr>
    <tr>
      <th>392</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.0</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>393</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>394</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.0</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>395</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.0</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>396</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.0</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>372 rows × 9 columns</p>
</div>

```python
print(df.info())
print(df.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 392 entries, 0 to 396
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           392 non-null    float64
     1   cylinders     392 non-null    int64  
     2   displacement  392 non-null    float64
     3   horsepower    392 non-null    float64
     4   weight        392 non-null    int64  
     5   acceleration  392 non-null    float64
     6   year          392 non-null    int64  
     7   origin        392 non-null    int64  
     8   name          392 non-null    object 
    dtypes: float64(4), int64(4), object(1)
    memory usage: 30.6+ KB
    None
                  mpg   cylinders  displacement  horsepower       weight  \
    count  392.000000  392.000000    392.000000  392.000000   392.000000   
    mean    23.445918    5.471939    194.411990  104.469388  2977.584184   
    std      7.805007    1.705783    104.644004   38.491160   849.402560   
    min      9.000000    3.000000     68.000000   46.000000  1613.000000   
    25%     17.000000    4.000000    105.000000   75.000000  2225.250000   
    50%     22.750000    4.000000    151.000000   93.500000  2803.500000   
    75%     29.000000    8.000000    275.750000  126.000000  3614.750000   
    max     46.600000    8.000000    455.000000  230.000000  5140.000000   
    
           acceleration        year      origin  
    count    392.000000  392.000000  392.000000  
    mean      15.541327   75.979592    1.576531  
    std        2.758864    3.683737    0.805518  
    min        8.000000   70.000000    1.000000  
    25%       13.775000   73.000000    1.000000  
    50%       15.500000   76.000000    1.000000  
    75%       17.025000   79.000000    2.000000  
    max       24.800000   82.000000    3.000000  

```python
plt.scatter(df["horsepower"], df["mpg"])
```

    <matplotlib.collections.PathCollection at 0x7482ad90a9c0>
    
![png](Linear%20Regression%20sklearn_files/Linear%20Regression%20sklearn_4_1.png)

```python
X = pd.DataFrame({"intercept": np.ones(df.shape[0]), "horsepower": df["horsepower"]})
X.head()
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
      <th>intercept</th>
      <th>horsepower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>1.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
model = sm.OLS(df["mpg"], X)
result = model.fit()
summarize(result)
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
      <th>coef</th>
      <th>std err</th>
      <th>t</th>
      <th>P&gt;|t|</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>intercept</th>
      <td>41.1259</td>
      <td>0.758</td>
      <td>54.222</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.1702</td>
      <td>0.007</td>
      <td>-24.281</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
result.params
```

    intercept     41.125860
    horsepower    -0.170189
    dtype: float64

```python
fig, ax = plt.subplots()
ax.scatter(df["horsepower"], df["mpg"], s=5)

coef = result.params
xlim = ax.get_xlim()
ylim = (coef[0] + coef[1] * xlim[0], coef[0] + coef[1] * xlim[1])
ax.plot(xlim, ylim, 'r-')
ax.set_title(coef)
```

    /tmp/ipykernel_152811/4053741418.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      ylim = (coef[0] + coef[1] * xlim[0], coef[0] + coef[1] * xlim[1])

    Text(0.5, 1.0, 'intercept     41.125860\nhorsepower    -0.170189\ndtype: float64')
    
![png](Linear%20Regression%20sklearn_files/Linear%20Regression%20sklearn_8_2.png)

```python
# confidence interval
params = np.array(result.params)
stderr = np.array(summarize(result)['std err'])
def conf_interval(params, stderr, x):
    low = np.dot((params - stderr), x)
    high = np.dot((params + stderr), x)
    return low, high
    
conf_interval(params, stderr, [1, 98])
```

    (23.003374498935447, 25.891374498935456)

```python
pred = result.get_prediction([1, 98])
pred.conf_int(alpha=0.05)
```

    array([[23.947671, 24.947078]])

```python
ax = plt.subplots(figsize=(8,8))[1]
ax.scatter(result.fittedvalues, result.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');
```
    
![png](Linear%20Regression%20sklearn_files/Linear%20Regression%20sklearn_11_0.png)

```python
df = df.drop(columns=['name'])
df.corr()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>-0.768159</td>
      <td>-0.803673</td>
      <td>-0.783845</td>
      <td>-0.832270</td>
      <td>0.383589</td>
      <td>0.568716</td>
      <td>0.555810</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>-0.768159</td>
      <td>1.000000</td>
      <td>0.953304</td>
      <td>0.842559</td>
      <td>0.902638</td>
      <td>-0.449479</td>
      <td>-0.296280</td>
      <td>-0.554618</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>-0.803673</td>
      <td>0.953304</td>
      <td>1.000000</td>
      <td>0.884850</td>
      <td>0.946904</td>
      <td>-0.475429</td>
      <td>-0.315380</td>
      <td>-0.612197</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.783845</td>
      <td>0.842559</td>
      <td>0.884850</td>
      <td>1.000000</td>
      <td>0.880128</td>
      <td>-0.641154</td>
      <td>-0.370743</td>
      <td>-0.448932</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.832270</td>
      <td>0.902638</td>
      <td>0.946904</td>
      <td>0.880128</td>
      <td>1.000000</td>
      <td>-0.381225</td>
      <td>-0.297841</td>
      <td>-0.578263</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.383589</td>
      <td>-0.449479</td>
      <td>-0.475429</td>
      <td>-0.641154</td>
      <td>-0.381225</td>
      <td>1.000000</td>
      <td>0.205426</td>
      <td>0.179090</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.568716</td>
      <td>-0.296280</td>
      <td>-0.315380</td>
      <td>-0.370743</td>
      <td>-0.297841</td>
      <td>0.205426</td>
      <td>1.000000</td>
      <td>0.160888</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0.555810</td>
      <td>-0.554618</td>
      <td>-0.612197</td>
      <td>-0.448932</td>
      <td>-0.578263</td>
      <td>0.179090</td>
      <td>0.160888</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
X = MS(df.drop(columns=['mpg'])).fit_transform(df)
```

```python
model = sm.OLS(df['mpg'], X)
result2 = model.fit()
summarize(result2)
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
      <th>coef</th>
      <th>std err</th>
      <th>t</th>
      <th>P&gt;|t|</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>intercept</th>
      <td>-19.7169</td>
      <td>4.963</td>
      <td>-3.973</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>-0.4081</td>
      <td>0.349</td>
      <td>-1.170</td>
      <td>0.243</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>0.0168</td>
      <td>0.009</td>
      <td>1.963</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.0174</td>
      <td>0.015</td>
      <td>-1.159</td>
      <td>0.247</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.0063</td>
      <td>0.001</td>
      <td>-8.213</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.1110</td>
      <td>0.105</td>
      <td>1.059</td>
      <td>0.290</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.7734</td>
      <td>0.054</td>
      <td>14.294</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>1.3909</td>
      <td>0.285</td>
      <td>4.886</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>

```python
anova_lm(result, result2)
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
      <th>df_resid</th>
      <th>ssr</th>
      <th>df_diff</th>
      <th>ss_diff</th>
      <th>F</th>
      <th>Pr(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>370.0</td>
      <td>8778.242950</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>364.0</td>
      <td>4119.951551</td>
      <td>6.0</td>
      <td>4658.291399</td>
      <td>68.593771</td>
      <td>7.767048e-57</td>
    </tr>
  </tbody>
</table>
</div>

```python
ax = plt.subplots(figsize=(8,8))[1]
ax.scatter(result2.fittedvalues, result2.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');
```
    
![png](Linear%20Regression%20sklearn_files/Linear%20Regression%20sklearn_16_0.png)

```python
vals = [VIF(X, i)
        for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif':vals},
                   index=X.columns[1:])
vif
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
      <th>vif</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cylinders</th>
      <td>11.149785</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>23.969825</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>9.578183</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>14.029209</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>2.414266</td>
    </tr>
    <tr>
      <th>year</th>
      <td>1.177402</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>1.745033</td>
    </tr>
  </tbody>
</table>
</div>

```python
infl = result2.get_influence ()
ax = plt.subplots(figsize =(8 ,8))[1]
ax.scatter(np.arange(X.shape [0]), infl.hat_matrix_diag)
ax.set_xlabel('Index ')
ax.set_ylabel('Leverage ')
np.argmax(infl.hat_matrix_diag)
```

    8
    
![png](Linear%20Regression%20sklearn_files/Linear%20Regression%20sklearn_18_1.png)

```python
df.describe()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>372.000000</td>
      <td>372.000000</td>
      <td>372.000000</td>
      <td>372.000000</td>
      <td>372.000000</td>
      <td>372.000000</td>
      <td>372.000000</td>
      <td>372.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.760215</td>
      <td>5.384409</td>
      <td>188.114247</td>
      <td>102.037634</td>
      <td>2956.629032</td>
      <td>15.738441</td>
      <td>76.301075</td>
      <td>1.594086</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.833502</td>
      <td>1.672699</td>
      <td>99.994614</td>
      <td>36.079109</td>
      <td>850.155868</td>
      <td>2.590570</td>
      <td>3.502898</td>
      <td>0.810489</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>68.000000</td>
      <td>46.000000</td>
      <td>1613.000000</td>
      <td>9.500000</td>
      <td>70.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.600000</td>
      <td>4.000000</td>
      <td>100.250000</td>
      <td>75.000000</td>
      <td>2219.750000</td>
      <td>14.000000</td>
      <td>73.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>4.000000</td>
      <td>141.000000</td>
      <td>91.500000</td>
      <td>2750.000000</td>
      <td>15.500000</td>
      <td>76.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.575000</td>
      <td>6.000000</td>
      <td>258.000000</td>
      <td>115.250000</td>
      <td>3581.750000</td>
      <td>17.300000</td>
      <td>79.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>230.000000</td>
      <td>5140.000000</td>
      <td>24.800000</td>
      <td>82.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
result2.rsquared
```

    0.8190301157401636

```python
result.rsquared
```

    0.6144135213369303

```python

```

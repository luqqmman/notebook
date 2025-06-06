```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
```


```python
data = fetch_california_housing()
```


```python
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target
```


```python
df = df.head(10000)
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
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
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
      <th>9995</th>
      <td>4.0775</td>
      <td>10.0</td>
      <td>6.140900</td>
      <td>1.025440</td>
      <td>1275.0</td>
      <td>2.495108</td>
      <td>39.14</td>
      <td>-121.03</td>
      <td>1.645</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>4.0848</td>
      <td>8.0</td>
      <td>6.350394</td>
      <td>1.091864</td>
      <td>1977.0</td>
      <td>2.594488</td>
      <td>39.13</td>
      <td>-121.07</td>
      <td>1.559</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>3.6333</td>
      <td>7.0</td>
      <td>7.243455</td>
      <td>1.107330</td>
      <td>1143.0</td>
      <td>2.992147</td>
      <td>39.11</td>
      <td>-121.05</td>
      <td>1.702</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>3.4630</td>
      <td>8.0</td>
      <td>6.363636</td>
      <td>1.166297</td>
      <td>1307.0</td>
      <td>2.898004</td>
      <td>39.08</td>
      <td>-121.04</td>
      <td>2.017</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>3.0781</td>
      <td>7.0</td>
      <td>5.487500</td>
      <td>1.050000</td>
      <td>246.0</td>
      <td>3.075000</td>
      <td>39.09</td>
      <td>-121.00</td>
      <td>1.625</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 9 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 9 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   MedInc      10000 non-null  float64
     1   HouseAge    10000 non-null  float64
     2   AveRooms    10000 non-null  float64
     3   AveBedrms   10000 non-null  float64
     4   Population  10000 non-null  float64
     5   AveOccup    10000 non-null  float64
     6   Latitude    10000 non-null  float64
     7   Longitude   10000 non-null  float64
     8   Target      10000 non-null  float64
    dtypes: float64(9)
    memory usage: 703.3 KB



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
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.718040</td>
      <td>31.847100</td>
      <td>5.212102</td>
      <td>1.090389</td>
      <td>1395.588700</td>
      <td>3.061855</td>
      <td>35.493820</td>
      <td>-119.472328</td>
      <td>2.04949</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.916912</td>
      <td>11.821967</td>
      <td>2.752832</td>
      <td>0.547035</td>
      <td>1090.838717</td>
      <td>6.098183</td>
      <td>1.959545</td>
      <td>1.808913</td>
      <td>1.16595</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.499900</td>
      <td>1.000000</td>
      <td>0.846154</td>
      <td>0.500000</td>
      <td>3.000000</td>
      <td>0.750000</td>
      <td>32.670000</td>
      <td>-124.350000</td>
      <td>0.14999</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.411900</td>
      <td>23.000000</td>
      <td>4.253385</td>
      <td>1.007078</td>
      <td>779.750000</td>
      <td>2.452830</td>
      <td>34.010000</td>
      <td>-121.590000</td>
      <td>1.17975</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.328900</td>
      <td>33.000000</td>
      <td>5.031476</td>
      <td>1.049645</td>
      <td>1137.500000</td>
      <td>2.851168</td>
      <td>34.170000</td>
      <td>-118.410000</td>
      <td>1.76600</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.544825</td>
      <td>40.000000</td>
      <td>5.830935</td>
      <td>1.097466</td>
      <td>1687.000000</td>
      <td>3.373184</td>
      <td>37.630000</td>
      <td>-118.210000</td>
      <td>2.58025</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.000100</td>
      <td>52.000000</td>
      <td>141.909091</td>
      <td>34.066667</td>
      <td>28566.000000</td>
      <td>599.714286</td>
      <td>41.950000</td>
      <td>-114.550000</td>
      <td>5.00001</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df.head(1000), hue="Target", diag_kind="kde")
plt.show()
```


    
![png](Linear%20Regression%20from%20scratch_files/Linear%20Regression%20from%20scratch_6_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()
```


    
![png](Linear%20Regression%20from%20scratch_files/Linear%20Regression%20from%20scratch_7_0.png)
    



```python
class LinearRegression:
    def __init__(self, lmbd=0.1, lr=0.1):
        self.lmbd = lmbd
        self.theta = 0
        self.theta_0 = 0
        self.lr = lr
        
    def gradient_fit(self, X_train, Y_train, max_iter=1000):
        self.theta = np.zeros(X_train.shape[1])

        history = []
        for _ in range(max_iter):
            pred = np.dot(X_train, self.theta) + self.theta_0
            err = Y_train - pred
            gradient_theta_0 = -np.mean(err)
            gradient_theta = -np.dot(X_train.T, err) / X_train.shape[0] + self.lmbd * self.theta

            self.theta_0 -= self.lr * gradient_theta_0
            self.theta -= self.lr * gradient_theta
            # history.append([self.theta_0] + self.theta)
        return self.theta_0, self.theta

    def closed_form_fit(self, X_train, Y_train):
        X = np.hstack([np.ones((X_train.shape[0], 1)), X_train]) 
        XTX = np.matmul(X.T, X)
        R = np.identity(XTX.shape[0]) * self.lmbd
        R[0, 0] = 0
        inv = np.linalg.inv(XTX + R)
        XTY = np.matmul(X.T, Y_train)
        theta = np.matmul(inv, XTY)
        self.theta_0 = theta[:1]
        self.theta = theta[1:]
        return self.theta_0, self.theta
        
    def predict(self, X_test):
        pred = np.dot(X_test, self.theta) + self.theta_0
        return pred
        # print(Y_test - pred)
        # return np.mean(np.square(Y_test - pred))

    def MSE(self, X_test, Y_test):
        # print(Y_test - pred)
        return np.mean(np.square(Y_test - self.predict(X_test)))

    def score(self, X_train, Y_train):
        pred = self.predict(X_train)
        TSS = np.sum(np.square(Y_train - np.mean(Y_train)))
        RSS = np.sum(np.square(pred - Y_train))
        return 1 - RSS/TSS

    def residual_plot(self, X_test, Y_test):
        pred = self.predict(X_test)
        residuals = Y_test - pred

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pred, residuals, color="blue", alpha=0.5, label="Residuals", s=1)
        ax.axhline(y=0, color='red', linestyle="--") 
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.legend()
        plt.show()
```


```python
# X = df[["MedInc"]]
X = df.drop(columns="Target")
Y = df["Target"]
```


```python
X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()
```


```python
train = X.index % 2 == 0
test = X.index % 2 == 1
X_train = X[train]
Y_train = Y[train]
X_test = X[test]
Y_test = Y[test]
```


```python
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
```


```python
model = LinearRegression(0.1)
```


```python
model.closed_form_fit(X_train, Y_train)
print(model.MSE(X_test, Y_test))
print(model.score(X_train, Y_train))
model.residual_plot(X_test, Y_test)
```

    0.41151109506529826
    0.6059167915575893



    
![png](Linear%20Regression%20from%20scratch_files/Linear%20Regression%20from%20scratch_14_1.png)
    



```python
model.gradient_fit(X_train, Y_train)
model.MSE(X_test, Y_test)
print(model.MSE(X_test, Y_test))
print(model.score(X_train, Y_train))
```

    0.43518215741620114
    0.5710105260707274



```python

```

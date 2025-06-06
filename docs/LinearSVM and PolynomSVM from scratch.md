# Linear and Polynom SVM from scratch
We will create SVM using the loss function and do the gradient descend manually


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
```


```python
df = pd.read_csv("dataset/Iris.csv", index_col="Id")
```

## We use the iris dataset 
But to make things simple we only classify is it iris setosa or not, because iris versicolor and iris virginica is'nt linearly separable


```python
sns.pairplot(df, hue="Species")
plt.show()
```


    
![png](LinearSVM%20and%20PolynomSVM%20from%20scratch_files/LinearSVM%20and%20PolynomSVM%20from%20scratch_4_0.png)
    



```python
x = np.array(df.drop(columns=["Species", "SepalLengthCm", "SepalWidthCm"]))
y = np.array([s == 'Iris-setosa' for s in df["Species"]])
```


```python
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(x, y, stratify=y, random_state=7)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
```


```python
def perceptron(lx, ly, maxiter=1000):
    theta = np.zeros(len(lx[0]), dtype=float)
    theta_0 = 0

    history = []
    for _ in range(maxiter):
        count = 0
        for x, y in zip(lx, ly):
            y = 1 if bool(y) else -1
            z = y * np.dot(theta, x) + theta_0
            if z <= 0:
                theta += y * x
                theta_0 += y
                history.append(np.copy(theta))
                count += 1
        if count == 0:
            break
    return theta_0, theta
```


```python
theta_0, theta = perceptron(X_train, Y_train)
```


```python
count = 0
# X_test = scaler.transform(X_test)
for x, y in zip(X_test, Y_test):
    z = theta_0 + np.dot(x, theta)
    count += ((z > 0) and y)
    count += ((z < 0) and not y)
```


```python
count/len(Y_test)
```




    0.5263157894736842




```python
theta
```




    array([-0.1, -2.2])




```python
theta_0
```




    5




```python

```


```python
class LinearSVM:
    def __init__(self, lx, ly):
        self.theta = np.zeros(len(lx[0]), dtype=float)
        self.theta_0 = 0
        self.lmbd = 0.1
        self.lr = 0.001
        self.maxiter=1000
        self.lx = lx
        self.ly = ly

    def transform_data(self, func):
        self.lx = np.apply_along_axis(func, 1, self.lx)
        self.theta = np.zeros(len(self.lx[0]), dtype=float)

    def train(self):
        history = []
        for _ in range(self.maxiter):
            count = 0
            for x, y in zip(self.lx, self.ly):
                y = 1 if bool(y) else -1
                z = 1 - (y * (np.dot(self.theta, x) + self.theta_0))
                gradient_theta = self.dregterm_dtheta(x)
                gradient_theta_0 = 0
                if z > 0:
                    gradient_theta += self.dlossterm_dtheta(x, y)
                    gradient_theta_0 += self.dlossterm_dtheta_0(y)
                    count += 1
                self.theta -= gradient_theta * self.lr
                self.theta_0 -= gradient_theta_0 * self.lr

                history.append(np.copy(self.theta))
            # if count == 0:
            #     break
        # print(history)
        return self.theta_0, self.theta

    def dregterm_dtheta(self, x):
        return self.lmbd * self.theta

    def dlossterm_dtheta(self, x, y):
        return -y*x

    def dlossterm_dtheta_0(self, y):
        return -y

    def predict(self, lx_test, ly_test):
        count = 0
        for x, y in zip(lx_test, ly_test):
            z = self.theta_0 + np.dot(x, self.theta)
            count += ((z > 0) and y)
            count += ((z < 0) and not y)
        return count / len(lx_test)
        
    def plot(self):
        # Plot data points
        plt.scatter(self.lxx[:, 0], self.lxx[:, 1], c=self.lyy, cmap=plt.cm.Paired, edgecolors='k', label='Data Points')

        # Plot decision boundary
        x1 = np.linspace(min(self.lxx[:, 0]), max(self.lxx[:, 0]), 100)
        x2 = -(self.theta[0] * x1 + self.theta_0) / self.theta[1]
        plt.plot(x1, x2, 'k-', label='Decision Boundary')

        # Plot margin boundaries
        margin = 1 / np.linalg.norm(self.theta)
        x2_upper = x2 + margin / np.linalg.norm(self.theta)
        x2_lower = x2 - margin / np.linalg.norm(self.theta)
        plt.plot(x1, x2_upper, 'k--', label='Margin Boundary', color="red")
        plt.plot(x1, x2_lower, 'k--', color="red")

        # Add labels and legend
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Linear SVM Decision and Margin Boundaries')
        plt.legend()
        plt.show()

```


```python
s = LinearSVM(X_train, Y_train)
```


```python
s.train()
```




    (-0.8530000000000006, array([-0.83165633, -0.66812085]))




```python
s.predict(X_test, Y_test)
```




    np.float64(1.0)




```python
s.plot()
```

    /tmp/ipykernel_4050/4131259921.py:68: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string "k--" (-> color='k'). The keyword argument will take precedence.
      plt.plot(x1, x2_upper, 'k--', label='Margin Boundary', color="red")
    /tmp/ipykernel_4050/4131259921.py:69: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string "k--" (-> color='k'). The keyword argument will take precedence.
      plt.plot(x1, x2_lower, 'k--', color="red")



    
![png](LinearSVM%20and%20PolynomSVM%20from%20scratch_files/LinearSVM%20and%20PolynomSVM%20from%20scratch_18_1.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt

class PolynomSVM:
    def __init__(self, lx, ly):
        self.copy_lx = lx
        self.lx = np.apply_along_axis(self.polynom, 1, lx)
        self.theta = np.zeros(len(self.lx[0]), dtype=float)
        self.theta_0 = 0
        self.lmbd = 0.9
        self.lr = 0.001
        self.maxiter = 1000
        self.ly = ly

    @staticmethod
    def polynom(arr):
        s = np.sqrt(2)
        return np.array([s * arr[0], s * arr[1], arr[0] ** 2, arr[1] ** 2, s * arr[0] * arr[1], 1])

    def train(self):
        for _ in range(self.maxiter):
            for x, y in zip(self.lx, self.ly):
                y = 1 if bool(y) else -1
                z = 1 - (y * (np.dot(self.theta, x) + self.theta_0))
                gradient_theta = self.lmbd * self.theta
                gradient_theta_0 = 0
                if z > 0:
                    gradient_theta -= y * x
                    gradient_theta_0 -= y
                self.theta -= gradient_theta * self.lr
                self.theta_0 -= gradient_theta_0 * self.lr
        return self.theta_0, self.theta

    def predict(self, lx_test, ly_test):
        lx_test = np.apply_along_axis(self.polynom, 1, lx_test)
        predictions = np.dot(lx_test, self.theta) + self.theta_0
        correct_predictions = ((predictions > 0) == ly_test).sum()
        print((predictions>0).sum())
        print(len(lx_test))
        return correct_predictions / len(lx_test)

    def plot(self):
        x_min, x_max = self.copy_lx[:, 0].min() - 1, self.copy_lx[:, 0].max() + 1
        y_min, y_max = self.copy_lx[:, 1].min() - 1, self.copy_lx[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_transformed = np.apply_along_axis(self.polynom, 1, grid)
        zz = np.dot(grid_transformed, self.theta) + self.theta_0
        zz = zz.reshape(xx.shape)
        
        plt.contourf(xx, yy, zz, levels=[-1, 0, 1], alpha=0.2, colors=["blue", "black", "red"])
        plt.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["blue", "black", "red"], linestyles=["dashed", "solid", "dashed"])
        
        plt.scatter(self.copy_lx[:, 0], self.copy_lx[:, 1], c=self.ly, cmap=plt.cm.Paired, edgecolors='k')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Polynomial SVM Decision Boundary")
        plt.show()
```


```python
s = PolynomSVM(X_train, Y_train)
```


```python
s.train()
```




    (-0.8980000000000007,
     array([-0.29973561, -0.27545061,  0.1579069 ,  0.10840018,  0.19147607,
            -0.00051781]))




```python
s.plot()
```


    
![png](LinearSVM%20and%20PolynomSVM%20from%20scratch_files/LinearSVM%20and%20PolynomSVM%20from%20scratch_22_0.png)
    



```python
s.predict(X_test, Y_test)
```

    13
    38





    np.float64(1.0)



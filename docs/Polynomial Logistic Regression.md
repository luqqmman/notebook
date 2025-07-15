5. We have seen that we can fit an SVM with a non-linear kernel in order
to perform classification using a non-linear decision boundary. We will
now see that we can also obtain a non-linear decision boundary by
performing logistic regression using non-linear transformations of the
features
- Generate a data set with n = 500 and p = 2, such that the obser-
vations belong to two classes with a quadratic decision boundary
between them. For instance, you can do this as follows:
rng = np.random.default_rng (5)
x1 = rng.uniform(size =500) - 0.5
x2 = rng.uniform(size =500) - 0.5
y = x1**2 - x2**2 > 0
- Plot the observations, colored according to their class labels.
Your plot should display X1 on the x-axis, and X2 on the y-
axis.
- Fit a logistic regression model to the data, using X1 and X2 as
predictors.
    Apply this model to the training data in order to obtain a pre-
dicted class label for each training observation. Plot the ob-
servations, colored according to the predicted class labels. The
decision boundary should be linear.
- Now fit a logistic regression model to the data using non-linear
functions of X1 and X2 as predictors (e.g. X2
1 , X1 ×X2, log(X2),
and so forth).
 - Apply this model to the training data in order to obtain a pre-
dicted class label for each training observation. Plot the ob-
servations, colored according to the predicted class labels. The
decision boundary should be obviously non-linear. If it is not,
then repeat (a)–(e) until you come up with an example in which
the predicted class labels are obviously non-linear.
 - Fit a support vector classifier to the data with X1 and X2 as
predictors. Obtain a class prediction for each training observa-
tion. Plot the observations, colored according to the predicted
class labels.
- Fit a SVM using a non-linear kernel to the data. Obtain a class
prediction for each training observation. Plot the observations,
colored according to the predicted class labels.
- Comment on your results.

```python
import numpy as np 
import pandas as pd
import sklearn.model_selection as skm

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay
from matplotlib.pyplot import subplots
from ISLP import load_data
from ISLP.svm import plot as plot_svm
```

```python
rng = np.random.default_rng(0)
x1 = rng.standard_normal(500)
x2 = rng.standard_normal(500)

X = np.array([x1, x2]).T
y = X[:,0]**2 - X[:,1]>1
X[:,1][y] -= 1
```

```python
fig, ax = subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], c=y)
```

    <matplotlib.collections.PathCollection at 0x7038467794c0>
    
![png](Polynomial%20Logistic%20Regression_files/Polynomial%20Logistic%20Regression_3_1.png)

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, shuffle=True, stratify=y, random_state=0)
lr_linear = LogisticRegression()
lr_linear.fit(X_train, y_train)
lr_linear_pred = lr_linear.predict(X_test)
accuracy_score(y_test, lr_linear_pred)
```

    0.896

```python
fig, ax = subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], c=lr_linear.predict(X))
```

    <matplotlib.collections.PathCollection at 0x703846912420>
    
![png](Polynomial%20Logistic%20Regression_files/Polynomial%20Logistic%20Regression_5_1.png)

```python
X = np.array([x1, x2, x1*x2, x1**2, x2**2]).T
X[:,1][y] -= 1
y = X[:,0]**2 - X[:,1]>1

X_train, X_test, y_train, y_test = skm.train_test_split(X, y, shuffle=True, stratify=y, random_state=0)
lr_poly = LogisticRegression()
lr_poly.fit(X_train, y_train)
lr_poly_pred = lr_poly.predict(X_test)
accuracy_score(y_test, lr_poly_pred)
```

    1.0

```python
fig, ax = subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], c=lr_poly.predict(X))
```

    <matplotlib.collections.PathCollection at 0x703845e4c050>
    
![png](Polynomial%20Logistic%20Regression_files/Polynomial%20Logistic%20Regression_7_1.png)

```python

```

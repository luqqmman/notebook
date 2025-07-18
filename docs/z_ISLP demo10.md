```python
import pandas as pd
import numpy as np
import sklearn.model_selection as skm
from matplotlib.pyplot import subplots

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from ISLP import load_data
```

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import RMSprop

from torchmetrics import MeanAbsoluteError, R2Score
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy
from torchinfo import summary
from torchvision.io import read_image

from torchvision.datasets import CIFAR100, MNIST
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Resize, Normalize, CenterCrop, ToTensor
```

```python
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
```

```python
from ISLP.torch import SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers
from ISLP.torch.imdb import load_lookup, load_tensor, load_sparse, load_sequential
from glob import glob
import json
```

```python
hitters = load_data("Hitters")
hitters.head()
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
hitters = hitters.dropna()
hitters = pd.get_dummies(hitters, columns=hitters.select_dtypes(include=["category"]).columns, drop_first=True)
X = hitters.drop(columns=["Salary"])
y = hitters["Salary"]
```

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, shuffle=True, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```python
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)
np.fabs(lr_pred-y_test).mean()
```

    265.87464986111604

```python
n = X_train.shape[0]
lam_max = np.fabs(X_train.T.dot(y_train - y_train.mean())).max() / n
param_grid = {'alpha': np.exp(np.linspace (0, np.log (0.01) , 100))* lam_max}
kfold = skm.KFold(n_splits=10, shuffle=True, random_state=0)
```

```python
lasso = skm.GridSearchCV(Lasso(), param_grid=param_grid, cv=kfold, scoring="neg_mean_absolute_error")
lasso.fit(X_train, y_train)
```

<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}

/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=Lasso(),
             param_grid={&#x27;alpha&#x27;: array([216.42356927, 206.58678403, 197.19709587, 188.23418352,
       179.67864937, 171.51197745, 163.71649337, 156.27532608,
       149.17237132, 142.39225682, 135.92030899, 129.74252118,
       123.84552335, 118.21655318, 112.84342839, 107.71452041,
       102.81872922,  98.14545929,  93.684596...
         9.15307838,   8.73705684,   8.33994413,   7.96088079,
         7.59904648,   7.25365808,   6.92396813,   6.60926309,
         6.30886188,   6.02211438,   5.74839998,   5.48712633,
         5.23772797,   4.99966515,   4.77242265,   4.55550868,
         4.34845378,   4.15080984,   3.96214913,   3.78206334,
         3.61016272,   3.44607525,   3.28944582,   3.13993543,
         2.99722052,   2.86099222,   2.73095571,   2.60682955,
         2.48834513,   2.375246  ,   2.2672874 ,   2.16423569])},
             scoring=&#x27;neg_mean_absolute_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=Lasso(),
             param_grid={&#x27;alpha&#x27;: array([216.42356927, 206.58678403, 197.19709587, 188.23418352,
       179.67864937, 171.51197745, 163.71649337, 156.27532608,
       149.17237132, 142.39225682, 135.92030899, 129.74252118,
       123.84552335, 118.21655318, 112.84342839, 107.71452041,
       102.81872922,  98.14545929,  93.684596...
         9.15307838,   8.73705684,   8.33994413,   7.96088079,
         7.59904648,   7.25365808,   6.92396813,   6.60926309,
         6.30886188,   6.02211438,   5.74839998,   5.48712633,
         5.23772797,   4.99966515,   4.77242265,   4.55550868,
         4.34845378,   4.15080984,   3.96214913,   3.78206334,
         3.61016272,   3.44607525,   3.28944582,   3.13993543,
         2.99722052,   2.86099222,   2.73095571,   2.60682955,
         2.48834513,   2.375246  ,   2.2672874 ,   2.16423569])},
             scoring=&#x27;neg_mean_absolute_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Lasso</div></div></label><div class="sk-toggleable__content fitted"><pre>Lasso(alpha=9.153078381962859)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Lasso</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Lasso.html">?<span>Documentation for Lasso</span></a></div></label><div class="sk-toggleable__content fitted"><pre>Lasso(alpha=9.153078381962859)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
lasso_pred = lasso.best_estimator_.predict(X_test)
np.fabs(lasso_pred-y_test).mean()
```

    254.4028106605372

```python
y.describe()
```

    count     263.000000
    mean      535.925882
    std       451.118681
    min        67.500000
    25%       190.000000
    50%       425.000000
    75%       750.000000
    max      2460.000000
    Name: Salary, dtype: float64

```python
class HittersModel(nn.Module):
    def __init__(self, input_size):
        super(HittersModel, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(50, 1)
        )

    def forward(self, X):
        X = self.flatten(X)
        return torch.flatten(self.sequential(X))
```

```python
hitters_model = HittersModel(X_train.shape[1])
summary(hitters_model)
```

    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    HittersModel                             --
    ├─Flatten: 1-1                           --
    ├─Sequential: 1-2                        --
    │    └─Linear: 2-1                       1,000
    │    └─ReLU: 2-2                         --
    │    └─Dropout: 2-3                      --
    │    └─Linear: 2-4                       51
    =================================================================
    Total params: 1,051
    Trainable params: 1,051
    Non-trainable params: 0
    =================================================================

```python
X_train_t = torch.tensor(X_train.astype(np.float32))
y_train_t = torch.tensor(y_train.to_numpy().astype(np.float32))
hit_train = TensorDataset(X_train_t, y_train_t)

X_test_t = torch.tensor(X_test.astype(np.float32))
y_test_t = torch.tensor(y_test.to_numpy().astype(np.float32))
hit_test = TensorDataset(X_test_t, y_test_t)
```

```python
hit_dm = SimpleDataModule(hit_train, hit_test, batch_size=32, num_workers=min(4, rec_num_workers()), validation=hit_test)
hit_module = SimpleModule.regression(hitters_model, metrics ={'mae':MeanAbsoluteError()})
hit_logger = CSVLogger('logs', name='hitters')
```

```python
hit_trainer = Trainer(deterministic=True, max_epochs=50, log_every_n_steps=5, logger=hit_logger, callbacks=[ErrorTracker()])
hit_trainer.fit(hit_module , datamodule=hit_dm)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    
      | Name  | Type         | Params | Mode 
    -----------------------------------------------
    0 | model | HittersModel | 1.1 K  | train
    1 | loss  | MSELoss      | 0      | train
    -----------------------------------------------
    1.1 K     Trainable params
    0         Non-trainable params
    1.1 K     Total params
    0.004     Total estimated model params size (MB)
    8         Modules in train mode
    0         Modules in eval mode

    `Trainer.fit` stopped: `max_epochs=50` reached.

```python
hit_trainer.test(hit_module, datamodule=hit_dm)
```

    Testing: |                                                | 0/? [00:00<?, ?it/s]

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_loss              135539.171875
            test_mae             271.5210876464844
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    [{'test_loss': 135539.171875, 'test_mae': 271.5210876464844}]

```python
hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)
```

```python
def summary_plot(results, ax, col='loss', valid_legend='Validation', training_legend='Training', ylabel='Loss', fontsize =20):
    for (column, color , label) in zip([f'train_{col}_epoch', f'valid_{col}'], ['black','red'], [training_legend, valid_legend]):
        results.plot(x='epoch', y=column, label=label, marker='o', color=color , ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        return ax
```

```python
fig , ax = subplots (1, 1, figsize =(6, 6))
ax = summary_plot(hit_results, ax , col='mae', ylabel='MAE', valid_legend='Validation (=Test)')
ax.set_ylim ([0, 400])
ax.set_xticks(np.linspace (0, 50, 11).astype(int));
```
    
![png](z_ISLP%20demo10_files/z_ISLP%20demo10_20_0.png)

```python
hitters_model.eval()
preds = hit_module(X_test_t)
torch.abs(y_test_t - preds).mean()
```

    tensor(271.5211, grad_fn=<MeanBackward0>)

```python
(mnist_train, mnist_test) = [MNIST(root='data', train=train, download=True, transform=ToTensor()) for train in [True, False]]
```

    100%|███████████████████████████████████████| 9.91M/9.91M [00:12<00:00, 766kB/s]
    100%|██████████████████████████████████████| 28.9k/28.9k [00:00<00:00, 70.4kB/s]
    100%|███████████████████████████████████████| 1.65M/1.65M [00:07<00:00, 220kB/s]
    100%|██████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 9.59MB/s]

```python
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.val_acc = MulticlassAccuracy(num_classes=10)
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.model = nn.Sequential(
            self.layer1, 
            self.layer2,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False,on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, train, test, batch_size=32, validation=None):
        super().__init__()
        self.train = train
        self.test = test
        self.val = validation or self.test
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

```

```python
mnist_dm = MNISTDataModule(mnist_train, mnist_test, batch_size=256)
mnist_model = MNISTModel()
mnist_logger = CSVLogger('logs', name='MNIST')
```

```python
for idx , (X_ ,Y_) in enumerate(mnist_dm.train_dataloader()):
    print('X: ', X_.shape)
    print('Y: ', Y_.shape)
    if idx >= 1:
        break

summary(mnist_model, input_data=X_, col_names=['input_size', 'output_size', 'num_params'])
```

    X:  torch.Size([256, 1, 28, 28])
    Y:  torch.Size([256])
    X:  torch.Size([256, 1, 28, 28])
    Y:  torch.Size([256])

    ===================================================================================================================
    Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
    ===================================================================================================================
    MNISTModel                               [256, 1, 28, 28]          [256, 10]                 --
    ├─Sequential: 1-1                        [256, 1, 28, 28]          [256, 10]                 --
    │    └─Sequential: 2-1                   [256, 1, 28, 28]          [256, 256]                --
    │    │    └─Flatten: 3-1                 [256, 1, 28, 28]          [256, 784]                --
    │    │    └─Linear: 3-2                  [256, 784]                [256, 256]                200,960
    │    │    └─ReLU: 3-3                    [256, 256]                [256, 256]                --
    │    │    └─Dropout: 3-4                 [256, 256]                [256, 256]                --
    │    └─Sequential: 2-2                   [256, 256]                [256, 128]                --
    │    │    └─Linear: 3-5                  [256, 256]                [256, 128]                32,896
    │    │    └─ReLU: 3-6                    [256, 128]                [256, 128]                --
    │    │    └─Dropout: 3-7                 [256, 128]                [256, 128]                --
    │    └─Linear: 2-3                       [256, 128]                [256, 10]                 1,290
    ===================================================================================================================
    Total params: 235,146
    Trainable params: 235,146
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 60.20
    ===================================================================================================================
    Input size (MB): 0.80
    Forward/backward pass size (MB): 0.81
    Params size (MB): 0.94
    Estimated Total Size (MB): 2.55
    ===================================================================================================================

```python
mnist_trainer = Trainer(deterministic=True, max_epochs=30, logger=mnist_logger)
mnist_trainer.fit(mnist_model, datamodule=mnist_dm)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    
      | Name      | Type               | Params | Mode 
    ---------------------------------------------------------
    0 | loss_fn   | CrossEntropyLoss   | 0      | train
    1 | train_acc | MulticlassAccuracy | 0      | train
    2 | val_acc   | MulticlassAccuracy | 0      | train
    3 | layer1    | Sequential         | 200 K  | train
    4 | layer2    | Sequential         | 32.9 K | train
    5 | model     | Sequential         | 235 K  | train
    ---------------------------------------------------------
    235 K     Trainable params
    0         Non-trainable params
    235 K     Total params
    0.941     Total estimated model params size (MB)
    14        Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:425: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:425: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.

    `Trainer.fit` stopped: `max_epochs=30` reached.

```python

```

```python
mnist_results = pd.read_csv(mnist_logger.experiment.metrics_file_path)
fig, ax = subplots()
mnist_results.dropna(subset=['val_acc']).plot(x='epoch', y='val_acc', ax=ax)
mnist_results.dropna(subset=['train_acc']).plot(x='epoch', y='train_acc', ax=ax)
```

    <Axes: xlabel='epoch'>
    
![png](z_ISLP%20demo10_files/z_ISLP%20demo10_29_1.png)

```python
cifar_train, cifar_test = [CIFAR100(root='data', train=train, download=True) for train in [True, False]]
```

```python
cifar_train.data.shape
```

    (50000, 32, 32, 3)

```python
transform = ToTensor()
cifar_train_X = torch.stack ([ transform(x) for x in cifar_train.data])
cifar_test_X = torch.stack ([ transform(x) for x in cifar_test.data])
cifar_train = TensorDataset(cifar_train_X, torch.tensor(cifar_train.targets))
cifar_test = TensorDataset(cifar_test_X, torch.tensor(cifar_test.targets))
```

```python
class BuildingBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=(3, 3), padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        return self.model(x)

class CIFARModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=100)
        self.learning_rate = 0.001
        self.conv_sizes = [
            (3, 32), # (32, 16, 16)
            (32, 64), # (64, 8, 8)
            (64, 128), # (128, 4, 4)
            (128, 256) # (256, 2, 2)
        ]
        self.conv = nn.Sequential(
            *[BuildingBlock(in_channel, out_channel) for in_channel, out_channel in self.conv_sizes]
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2*2*256, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        conv = self.conv(x)
        out = self.dense(conv)
        return out

    def step(self, batch, batch_idx, t='train'):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        self.log(f'{t}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{t}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, train_t, val_t, batch_size=32):
        super().__init__()
        self.train_t = train_t
        self.val_t = val_t
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_t, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_t, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return self.val_dataloader(batch_size=self.batch_size)
```

```python
cifar_model = CIFARModel()
cifar_dm = CIFARDataModule(cifar_train, cifar_test, batch_size=128)
cifar_logger = CSVLogger('logs', name='CIFAR100')
cifar_trainer = Trainer(max_epochs=30, logger=cifar_logger)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
cifar_trainer.fit(cifar_model, datamodule=cifar_dm)
```
    
      | Name     | Type               | Params | Mode 
    --------------------------------------------------------
    0 | loss_fn  | CrossEntropyLoss   | 0      | train
    1 | accuracy | MulticlassAccuracy | 0      | train
    2 | conv     | Sequential         | 388 K  | train
    3 | dense    | Sequential         | 576 K  | train
    --------------------------------------------------------
    964 K     Trainable params
    0         Non-trainable params
    964 K     Total params
    3.858     Total estimated model params size (MB)
    29        Modules in train mode
    0         Modules in eval mode

    `Trainer.fit` stopped: `max_epochs=30` reached.

```python
fig, ax = subplots()
cifar_results = pd.read_csv(cifar_logger.experiment.metrics_file_path)
cifar_results.dropna(subset=['val_accuracy']).plot(x='epoch', y='val_accuracy', ax=ax)
cifar_results.dropna(subset=['train_accuracy']).plot(x='epoch', y='train_accuracy', ax=ax)
```

    <Axes: xlabel='epoch'>
    
![png](z_ISLP%20demo10_files/z_ISLP%20demo10_36_1.png)

```python
for i, (X_, Y_) in enumerate(cifar_dm.train_dataloader()):
    # print(X_)
    # print(Y_)
    if i >= 1:
        break
        
cifar_model = CIFARModel ()
summary(cifar_model,input_data=X_ , col_names =['input_size','output_size', 'num_params'])
```

    ===================================================================================================================
    Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
    ===================================================================================================================
    CIFARModel                               [32, 3, 32, 32]           [32, 100]                 --
    ├─Sequential: 1-1                        [32, 3, 32, 32]           [32, 256, 2, 2]           --
    │    └─BuildingBlock: 2-1                [32, 3, 32, 32]           [32, 32, 16, 16]          --
    │    │    └─Sequential: 3-1              [32, 3, 32, 32]           [32, 32, 16, 16]          896
    │    └─BuildingBlock: 2-2                [32, 32, 16, 16]          [32, 64, 8, 8]            --
    │    │    └─Sequential: 3-2              [32, 32, 16, 16]          [32, 64, 8, 8]            18,496
    │    └─BuildingBlock: 2-3                [32, 64, 8, 8]            [32, 128, 4, 4]           --
    │    │    └─Sequential: 3-3              [32, 64, 8, 8]            [32, 128, 4, 4]           73,856
    │    └─BuildingBlock: 2-4                [32, 128, 4, 4]           [32, 256, 2, 2]           --
    │    │    └─Sequential: 3-4              [32, 128, 4, 4]           [32, 256, 2, 2]           295,168
    ├─Sequential: 1-2                        [32, 256, 2, 2]           [32, 100]                 --
    │    └─Flatten: 2-5                      [32, 256, 2, 2]           [32, 1024]                --
    │    └─Dropout: 2-6                      [32, 1024]                [32, 1024]                --
    │    └─Linear: 2-7                       [32, 1024]                [32, 512]                 524,800
    │    └─ReLU: 2-8                         [32, 512]                 [32, 512]                 --
    │    └─Linear: 2-9                       [32, 512]                 [32, 100]                 51,300
    ===================================================================================================================
    Total params: 964,516
    Trainable params: 964,516
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 501.70
    ===================================================================================================================
    Input size (MB): 0.39
    Forward/backward pass size (MB): 15.89
    Params size (MB): 3.86
    Estimated Total Size (MB): 20.14
    ===================================================================================================================

```python
max_num_workers =10
(imdb_train, imdb_test) = load_tensor(root='data/IMDB')
imdb_dm = SimpleDataModule(imdb_train, imdb_test, validation =2000, num_workers=min(6, max_num_workers), batch_size =512)
```

    Retrieving "IMDB_X_test.tensor.gz" from "http://imdb.jtaylor.su.domains/jtaylor/data/".
    Retrieving "IMDB_X_train.tensor.gz" from "http://imdb.jtaylor.su.domains/jtaylor/data/".
    Retrieving "IMDB_Y_test.npy" from "http://imdb.jtaylor.su.domains/jtaylor/data/".
    Retrieving "IMDB_Y_train.npy" from "http://imdb.jtaylor.su.domains/jtaylor/data/".

```python
class IMDBModel(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.learning_rate = 0.001
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)
        
    def step(self, batch, batch_idx, t='train'):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y.unsqueeze(1).float())
        acc = self.accuracy(preds, y.unsqueeze(1).float())
        self.log(f'{t}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{t}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, train_t, val_t, batch_size=32):
        super().__init__()
        self.train_t = train_t
        self.val_t = val_t
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_t, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_t, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return self.val_dataloader(batch_size=self.batch_size)
```

```python
imdb_model = IMDBModel(imdb_train.tensors[0].size()[1])
imdb_dm = IMDBDataModule(imdb_train, imdb_test)
imdb_logger = CSVLogger("logs", name="imdb")
imdb_trainer = Trainer(
    deterministic=True, 
    logger=imdb_logger, 
    max_epochs=30, 
    callbacks=[EarlyStopping(monitor='val_accuracy', mode='min', patience=5)]
)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
imdb_trainer.fit(imdb_model, datamodule=imdb_dm)
```
    
      | Name     | Type              | Params | Mode 
    -------------------------------------------------------
    0 | loss_fn  | BCEWithLogitsLoss | 0      | train
    1 | accuracy | BinaryAccuracy    | 0      | train
    2 | model    | Sequential        | 160 K  | train
    -------------------------------------------------------
    160 K     Trainable params
    0         Non-trainable params
    160 K     Total params
    0.641     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode

```python
fig, ax = subplots()
imdb_results = pd.read_csv(imdb_logger.experiment.metrics_file_path)
imdb_results.dropna(subset=['val_accuracy']).plot(x='epoch', y='val_accuracy', ax=ax)
imdb_results.dropna(subset=['train_accuracy']).plot(x='epoch', y='train_accuracy', ax=ax)
ax.set_ylim([0, 1.1])

```

    (0.0, 1.1)
    
![png](z_ISLP%20demo10_files/z_ISLP%20demo10_42_1.png)

```python
(imdb_seq_train ,
imdb_seq_test) = load_sequential(root='data/IMDB')
padded_sample = np.asarray(imdb_seq_train.tensors [0][0])
sample_review = padded_sample[padded_sample > 0][:12]
sample_review [:12]
```

    Retrieving "IMDB_S_train.tensor.gz" from "http://imdb.jtaylor.su.domains/jtaylor/data/".
    Retrieving "IMDB_S_test.tensor.gz" from "http://imdb.jtaylor.su.domains/jtaylor/data/".

    array([   1,   14,   22,   16,   43,  530,  973, 1622, 1385,   65,  458,
           4468], dtype=int32)

```python
lookup = load_lookup(root='data/IMDB')
' '.join(lookup[i] for i in sample_review)
```

    Retrieving "IMDB_word_index.pkl" from "http://imdb.jtaylor.su.domains/jtaylor/data/".

    "<START> this film was just brilliant casting location scenery story direction everyone's"

```python
imdb_seq_test.tensors[0][0]
```

    tensor([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               1,  591,  202,   14,   31,    6,  717,   10,   10,    2,    2,    5,
               4,  360,    7,    4,  177, 5760,  394,  354,    4,  123,    9, 1035,
            1035, 1035,   10,   10,   13,   92,  124,   89,  488, 7944,  100,   28,
            1668,   14,   31,   23,   27, 7479,   29,  220,  468,    8,  124,   14,
             286,  170,    8,  157,   46,    5,   27,  239,   16,  179,    2,   38,
              32,   25, 7944,  451,  202,   14,    6,  717], dtype=torch.int32)

```python
nn.Embedding?
```

    [0;31mInit signature:[0m
    [0mnn[0m[0;34m.[0m[0mEmbedding[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mnum_embeddings[0m[0;34m:[0m [0mint[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0membedding_dim[0m[0;34m:[0m [0mint[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpadding_idx[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mint[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmax_norm[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mfloat[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mnorm_type[0m[0;34m:[0m [0mfloat[0m [0;34m=[0m [0;36m2.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mscale_grad_by_freq[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msparse[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0m_weight[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mtorch[0m[0;34m.[0m[0mTensor[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0m_freeze[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdtype[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
    
    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    
    .. note::
        When :attr:`max_norm` is not ``None``, :class:`Embedding`'s forward method will modify the
        :attr:`weight` tensor in-place. Since tensors needed for gradient computations cannot be
        modified in-place, performing a differentiable operation on ``Embedding.weight`` before
        calling :class:`Embedding`'s forward method requires cloning ``Embedding.weight`` when
        :attr:`max_norm` is not ``None``. For example::
    
            n, d, m = 3, 5, 7
            embedding = nn.Embedding(n, d, max_norm=1.0)
            W = torch.randn((m, d), requires_grad=True)
            idx = torch.tensor([1, 2])
            a = embedding.weight.clone() @ W.t()  # weight must be cloned for this to be differentiable
            b = embedding(idx) @ W.t()  # modifies weight in-place
            out = (a.unsqueeze(0) + b.unsqueeze(1))
            loss = out.sigmoid().prod()
            loss.backward()
    
    Examples::
    
        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],
    
                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])
    
        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0, 2, 0, 5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])
    
        >>> # example of changing `pad` vector
        >>> padding_idx = 0
        >>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
        >>> with torch.no_grad():
        ...     embedding.weight[padding_idx] = torch.ones(3)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 1.0000,  1.0000,  1.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
    [0;31mInit docstring:[0m Initialize internal Module state, shared by both nn.Module and ScriptModule.
    [0;31mFile:[0m           ~/Lab/islp/venv/lib/python3.12/site-packages/torch/nn/modules/sparse.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     Embedding, Embedding

```python
nn.LSTM?
```

    [0;31mInit signature:[0m [0mnn[0m[0;34m.[0m[0mLSTM[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    __init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,proj_size=0,device=None,dtype=None)
    
    Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:
    
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}
    
    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.
    
    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.
    
    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.
    
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0
    
    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.
    
        where:
    
        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}
    
    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence. When ``bidirectional=True``, `output` will contain
          a concatenation of the forward and reverse hidden states at each time step in the sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence. When ``bidirectional=True``,
          `h_n` will contain a concatenation of the final forward and reverse hidden states, respectively.
        * **c_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the sequence. When ``bidirectional=True``,
          `c_n` will contain a concatenation of the final forward and reverse cell states, respectively.
    
    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If
            ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, proj_size)`.
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
        weight_hr_l[k] : the learnable projection weights of the :math:`\text{k}^{th}` layer
            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was
            specified.
        weight_ih_l[k]_reverse: Analogous to `weight_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hh_l[k]_reverse:  Analogous to `weight_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_ih_l[k]_reverse:  Analogous to `bias_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_hh_l[k]_reverse:  Analogous to `bias_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hr_l[k]_reverse:  Analogous to `weight_hr_l[k]` for the reverse direction.
            Only present when ``bidirectional=True`` and ``proj_size > 0`` was specified.
    
    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`
    
    .. note::
        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.
    
    .. note::
        For bidirectional LSTMs, `h_n` is not equivalent to the last element of `output`; the
        former contains the final forward and reverse hidden states, while the latter contains the
        final forward hidden state and the initial reverse hidden state.
    
    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.
    
    .. note::
        ``proj_size`` should be smaller than ``hidden_size``.
    
    .. include:: ../cudnn_rnn_determinism.rst
    
    .. include:: ../cudnn_persistent_rnn.rst
    
    Examples::
    
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    [0;31mInit docstring:[0m Initialize internal Module state, shared by both nn.Module and ScriptModule.
    [0;31mFile:[0m           ~/Lab/islp/venv/lib/python3.12/site-packages/torch/nn/modules/rnn.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     

```python
nn.*
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score

from ISLP import load_data , confusion_table
from ISLP.models import ModelSpec as MS
from ISLP.bart import BART
```

7. in Section 8.3.3, we applied random forests to the Boston data using
max_features = 6 and using n_estimators = 100 and n_estimators =
500 . Create a plot displaying the test error resulting from random
forests on this data set for a more comprehensive range of values
for max_features and n_estimators. You can model your plot after
Figure 8.10. Describe the results obtained.

```python
boston = load_data("Boston")
boston
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>5.33</td>
      <td>36.2</td>
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
    </tr>
    <tr>
      <th>501</th>
      <td>0.06263</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0</td>
      <td>0.573</td>
      <td>6.593</td>
      <td>69.1</td>
      <td>2.4786</td>
      <td>1</td>
      <td>273</td>
      <td>21.0</td>
      <td>9.67</td>
      <td>22.4</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.04527</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0</td>
      <td>0.573</td>
      <td>6.120</td>
      <td>76.7</td>
      <td>2.2875</td>
      <td>1</td>
      <td>273</td>
      <td>21.0</td>
      <td>9.08</td>
      <td>20.6</td>
    </tr>
    <tr>
      <th>503</th>
      <td>0.06076</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0</td>
      <td>0.573</td>
      <td>6.976</td>
      <td>91.0</td>
      <td>2.1675</td>
      <td>1</td>
      <td>273</td>
      <td>21.0</td>
      <td>5.64</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>504</th>
      <td>0.10959</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0</td>
      <td>0.573</td>
      <td>6.794</td>
      <td>89.3</td>
      <td>2.3889</td>
      <td>1</td>
      <td>273</td>
      <td>21.0</td>
      <td>6.48</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>0.04741</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0</td>
      <td>0.573</td>
      <td>6.030</td>
      <td>80.8</td>
      <td>2.5050</td>
      <td>1</td>
      <td>273</td>
      <td>21.0</td>
      <td>7.88</td>
      <td>11.9</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 13 columns</p>
</div>

```python
X = boston.drop(columns=["medv"])
y = boston["medv"]
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.5, shuffle=True, random_state=0)
```

```python
rf = RandomForestRegressor()
params = {'n_estimators': np.arange(50, 350, 50), 'max_features': [1, X_train.shape[1], "sqrt", "log2"]}
model = skm.GridSearchCV(rf, param_grid=params, verbose=1)
model.fit(X_train, y_train)
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits

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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(estimator=RandomForestRegressor(),
             param_grid={&#x27;max_features&#x27;: [1, 12, &#x27;sqrt&#x27;, &#x27;log2&#x27;],
                         &#x27;n_estimators&#x27;: array([ 50, 100, 150, 200, 250, 300])},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(estimator=RandomForestRegressor(),
             param_grid={&#x27;max_features&#x27;: [1, 12, &#x27;sqrt&#x27;, &#x27;log2&#x27;],
                         &#x27;n_estimators&#x27;: array([ 50, 100, 150, 200, 250, 300])},
             verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: RandomForestRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=&#x27;log2&#x27;, n_estimators=50)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=&#x27;log2&#x27;, n_estimators=50)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
df = pd.DataFrame(model.cv_results_["params"])
df["r2"] = model.cv_results_["mean_test_score"]
grp = df.groupby("max_features")
```

```python
_, ax = plt.subplots()
for name, group in df.groupby('max_features'):
    ax.plot(group["n_estimators"], group["r2"], label=name, marker='o')

plt.title('Mean Test Score vs. n_estimators (Grouped by max_features)')
plt.xlabel('n_estimators')
plt.ylabel('mean_test_score')
plt.legend(title='max_features')
plt.grid(True)
plt.tight_layout()
plt.show()

```
    
![png](Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_files/Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_6_0.png)

```python
pred = model.best_estimator_.predict(X_test)
r2_score(y_test, pred), mean_squared_error(y_test, pred)
```

    (0.7790723677094152, 16.749676711462456)

8. in the lab, a classification tree was applied to the Carseats data set af-
ter converting Sales into a qualitative response variable. Now we will
seek to predict Sales using regression trees and related approaches,
treating the response as a quantitative variable
- (a) Split the data set into a training set and a test set.
- (b) Fit a regression tree to the training set. Plot the tree, and inter-
pret the results. What test MSE do you obtain?
- (c) Use cross-validation in order to determine the optimal level of
tree complexity. Does pruning the tree improve the test MSE?
- (d) Use the bagging approach in order to analyze this data. What
- test MSE do you obtain? Use the feature_importance_ values to
determine which variables are most important.
- (e) Use random forests to analyze this data. What test MSE do
you obtain? Use the feature_importance_ values to determine
which variables are most important. Describe the effect of m, the
number of variables considered at each split, on the error rate
obtained.
- (f) Now analyze the data using BART, and report your results.

```python
carseats = load_data("Carseats")
carseats.head()
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
      <th>Sales</th>
      <th>CompPrice</th>
      <th>Income</th>
      <th>Advertising</th>
      <th>Population</th>
      <th>Price</th>
      <th>ShelveLoc</th>
      <th>Age</th>
      <th>Education</th>
      <th>Urban</th>
      <th>US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.50</td>
      <td>138</td>
      <td>73</td>
      <td>11</td>
      <td>276</td>
      <td>120</td>
      <td>Bad</td>
      <td>42</td>
      <td>17</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.22</td>
      <td>111</td>
      <td>48</td>
      <td>16</td>
      <td>260</td>
      <td>83</td>
      <td>Good</td>
      <td>65</td>
      <td>10</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.06</td>
      <td>113</td>
      <td>35</td>
      <td>10</td>
      <td>269</td>
      <td>80</td>
      <td>Medium</td>
      <td>59</td>
      <td>12</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.40</td>
      <td>117</td>
      <td>100</td>
      <td>4</td>
      <td>466</td>
      <td>97</td>
      <td>Medium</td>
      <td>55</td>
      <td>14</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.15</td>
      <td>141</td>
      <td>64</td>
      <td>3</td>
      <td>340</td>
      <td>128</td>
      <td>Bad</td>
      <td>38</td>
      <td>13</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>

```python
y = carseats["Sales"]
X = pd.get_dummies(carseats.drop(columns="Sales"), columns=["ShelveLoc", "Urban", "US"], drop_first=True)
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
      <th>CompPrice</th>
      <th>Income</th>
      <th>Advertising</th>
      <th>Population</th>
      <th>Price</th>
      <th>Age</th>
      <th>Education</th>
      <th>ShelveLoc_Good</th>
      <th>ShelveLoc_Medium</th>
      <th>Urban_Yes</th>
      <th>US_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>138</td>
      <td>73</td>
      <td>11</td>
      <td>276</td>
      <td>120</td>
      <td>42</td>
      <td>17</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>111</td>
      <td>48</td>
      <td>16</td>
      <td>260</td>
      <td>83</td>
      <td>65</td>
      <td>10</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113</td>
      <td>35</td>
      <td>10</td>
      <td>269</td>
      <td>80</td>
      <td>59</td>
      <td>12</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>117</td>
      <td>100</td>
      <td>4</td>
      <td>466</td>
      <td>97</td>
      <td>55</td>
      <td>14</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>141</td>
      <td>64</td>
      <td>3</td>
      <td>340</td>
      <td>128</td>
      <td>38</td>
      <td>13</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.5, random_state=0, shuffle=True)
kfold = skm.KFold(n_splits=10, shuffle=True, random_state=0)
```

```python
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
dtr_pred = dtr.predict(X_test)
mean_squared_error(y_test, dtr_pred), r2_score(y_test, dtr_pred)
```

    (6.007378000000001, 0.18420298947019487)

```python
ccp = dtr.cost_complexity_pruning_path(X_train, y_train)['ccp_alphas']
pruned = skm.GridSearchCV(dtr, param_grid={'ccp_alpha': ccp}, cv=kfold, scoring="neg_mean_squared_error")
pruned.fit(X_train, y_train)
```

<style>#sk-container-id-3 {
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

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
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

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
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

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
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

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
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

#sk-container-id-3 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
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

#sk-container-id-3 a.estimator_doc_link {
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

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=DecisionTreeRegressor(random_state=0),
             param_grid={&#x27;ccp_alpha&#x27;: array([0.00000000e+00, 2.50000000e-07, 1.00000000e-06, 1.00000000e-06,
       2.25000000e-06, 2.25000000e-06, 4.00000000e-06, 4.00000000e-06,
       9.00000000e-06, 9.00000000e-06, 9.00000000e-06, 9.00000000e-06,
       1.22500000e-05, 1.60000000e-0...
       6.04006667e-02, 6.37144802e-02, 6.85004135e-02, 7.09593750e-02,
       7.10987292e-02, 7.40173444e-02, 7.45467149e-02, 7.81200795e-02,
       1.05294704e-01, 1.27332857e-01, 1.65842886e-01, 1.81486201e-01,
       1.86376688e-01, 2.52891440e-01, 2.59843794e-01, 2.67189360e-01,
       3.46654183e-01, 3.76660787e-01, 6.13656686e-01, 6.43168488e-01,
       1.07979677e+00, 1.53159950e+00])},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=DecisionTreeRegressor(random_state=0),
             param_grid={&#x27;ccp_alpha&#x27;: array([0.00000000e+00, 2.50000000e-07, 1.00000000e-06, 1.00000000e-06,
       2.25000000e-06, 2.25000000e-06, 4.00000000e-06, 4.00000000e-06,
       9.00000000e-06, 9.00000000e-06, 9.00000000e-06, 9.00000000e-06,
       1.22500000e-05, 1.60000000e-0...
       6.04006667e-02, 6.37144802e-02, 6.85004135e-02, 7.09593750e-02,
       7.10987292e-02, 7.40173444e-02, 7.45467149e-02, 7.81200795e-02,
       1.05294704e-01, 1.27332857e-01, 1.65842886e-01, 1.81486201e-01,
       1.86376688e-01, 2.52891440e-01, 2.59843794e-01, 2.67189360e-01,
       3.46654183e-01, 3.76660787e-01, 6.13656686e-01, 6.43168488e-01,
       1.07979677e+00, 1.53159950e+00])},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: DecisionTreeRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeRegressor(ccp_alpha=0.07401734444444534, random_state=0)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeRegressor.html">?<span>Documentation for DecisionTreeRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeRegressor(ccp_alpha=0.07401734444444534, random_state=0)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
pruned_pred = pruned.best_estimator_.predict(X_test)
mean_squared_error(y_test, pruned_pred), r2_score(y_test, pruned_pred)
```

    (5.377231455417805, 0.2697763739427288)

```python
dtr.get_n_leaves(), pruned.best_estimator_.get_n_leaves()
```

    (200, 18)

```python
_, ax = plt.subplots()
ax.plot(ccp, pruned.cv_results_["mean_test_score"])
ax.axvline(pruned.best_params_["ccp_alpha"]), pruned.best_params_["ccp_alpha"]
```

    (<matplotlib.lines.Line2D at 0x7f6d054b07a0>, 0.07401734444444534)
    
![png](Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_files/Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_16_1.png)

```python
params = {"max_features": [X_train.shape[1], "sqrt", "log2"], "n_estimators": [50, 100, 200, 500]}
rf = skm.GridSearchCV(RandomForestRegressor(), param_grid=params, cv=kfold, scoring="neg_mean_squared_error")
rf.fit(X_train, y_train)
```

<style>#sk-container-id-5 {
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

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
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

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
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

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
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

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
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

#sk-container-id-5 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
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

#sk-container-id-5 a.estimator_doc_link {
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

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=RandomForestRegressor(),
             param_grid={&#x27;max_features&#x27;: [11, &#x27;sqrt&#x27;, &#x27;log2&#x27;],
                         &#x27;n_estimators&#x27;: [50, 100, 200, 500]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=RandomForestRegressor(),
             param_grid={&#x27;max_features&#x27;: [11, &#x27;sqrt&#x27;, &#x27;log2&#x27;],
                         &#x27;n_estimators&#x27;: [50, 100, 200, 500]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: RandomForestRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=11, n_estimators=500)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_features=11, n_estimators=500)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
rf_pred = rf.best_estimator_.predict(X_test)
mean_squared_error(y_test, rf_pred), r2_score(y_test, rf_pred)
```

    (2.656156527654005, 0.6392961197448597)

```python
df = pd.DataFrame(rf.cv_results_["params"])
df["neg_mean_squared_error"] = rf.cv_results_["mean_test_score"]
grp = df.groupby("max_features")

_, ax = plt.subplots()
for name, group in df.groupby("max_features"):
    ax.plot(group["n_estimators"], group["neg_mean_squared_error"], label=name, marker='o')

plt.legend()
plt.title('Negative MSE vs n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('neg mse')
plt.legend(title='max_features')
plt.grid(True)
plt.tight_layout()
plt.show()
```
    
![png](Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_files/Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_19_0.png)

```python
pd.DataFrame({"importance": rf.best_estimator_.feature_importances_}, index=X.columns).sort_values(by="importance", ascending=False)
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
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Price</th>
      <td>0.334300</td>
    </tr>
    <tr>
      <th>ShelveLoc_Good</th>
      <td>0.154064</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.113329</td>
    </tr>
    <tr>
      <th>CompPrice</th>
      <td>0.087943</td>
    </tr>
    <tr>
      <th>ShelveLoc_Medium</th>
      <td>0.080999</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>0.068796</td>
    </tr>
    <tr>
      <th>Population</th>
      <td>0.058244</td>
    </tr>
    <tr>
      <th>Advertising</th>
      <td>0.057147</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.032739</td>
    </tr>
    <tr>
      <th>US_Yes</th>
      <td>0.007296</td>
    </tr>
    <tr>
      <th>Urban_Yes</th>
      <td>0.005143</td>
    </tr>
  </tbody>
</table>
</div>

10. We now use boosting to predict Salary in the Hitters data set.
(a) Remove the observations for whom the salary information is
unknown, and then log-transform the salaries.
(b) Create a training set consisting of the first 200 observations, and
a test set consisting of the remaining observations.
(c) Perform boosting on the training set with 1,000 trees for a range
of values of the shrinkage parameter λ. Produce a plot with
different shrinkage values on the x-axis and the corresponding
training set MSE on the y-axis.
(d) Produce a plot with different shrinkage values on the x-axis and
the corresponding test set MSE on the y-axis.
(e) Compare the test MSE of boosting to the test MSE that results
from applying two of the regression approaches seen in
Chapters 3 and 6.
(f) Which variables appear to be the most important predictors in
the boosted model?
(g) Now apply bagging to the training set. What is the test set MSE
for this approach?

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
hitters.describe()
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
hitters = hitters.dropna()
X = pd.get_dummies(hitters.drop(columns=["Salary"]), columns=X.select_dtypes(include=["category"]).columns)
y = np.log(hitters["Salary"])

n_train = 200
X_train = X[:n_train]
X_test = X[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]
```

```python
alphas = 10**np.linspace(0, -5, 10)
_, ax = plt.subplots()
ax.plot(np.linspace(0, -5, 10), alphas, marker="o")
```

    [<matplotlib.lines.Line2D at 0x7f6d0393fad0>]
    
![png](Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_files/Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_25_1.png)

```python
gbr = skm.GridSearchCV(GradientBoostingRegressor(n_estimators=1000), param_grid={"learning_rate": alphas}, cv=kfold, scoring="neg_mean_squared_error")
gbr.fit(X_train, y_train)
```

<style>#sk-container-id-7 {
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

#sk-container-id-7 {
  color: var(--sklearn-color-text);
}

#sk-container-id-7 pre {
  padding: 0;
}

#sk-container-id-7 input.sk-hidden--visually {
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

#sk-container-id-7 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-7 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-7 div.sk-text-repr-fallback {
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

#sk-container-id-7 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-7 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-7 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-7 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-7 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-7 div.sk-serial {
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

#sk-container-id-7 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-7 label.sk-toggleable__label {
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

#sk-container-id-7 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-7 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-7 div.sk-label label.sk-toggleable__label,
#sk-container-id-7 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-7 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-7 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-7 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-7 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-7 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-7 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-7 div.sk-estimator.fitted:hover {
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

#sk-container-id-7 a.estimator_doc_link {
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

#sk-container-id-7 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-7 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-7 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=GradientBoostingRegressor(n_estimators=1000),
             param_grid={&#x27;learning_rate&#x27;: array([1.00000000e+00, 2.78255940e-01, 7.74263683e-02, 2.15443469e-02,
       5.99484250e-03, 1.66810054e-03, 4.64158883e-04, 1.29154967e-04,
       3.59381366e-05, 1.00000000e-05])},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=GradientBoostingRegressor(n_estimators=1000),
             param_grid={&#x27;learning_rate&#x27;: array([1.00000000e+00, 2.78255940e-01, 7.74263683e-02, 2.15443469e-02,
       5.99484250e-03, 1.66810054e-03, 4.64158883e-04, 1.29154967e-04,
       3.59381366e-05, 1.00000000e-05])},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: GradientBoostingRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor(learning_rate=0.005994842503189409, n_estimators=1000)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GradientBoostingRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor(learning_rate=0.005994842503189409, n_estimators=1000)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
gbr_pred = gbr.best_estimator_.predict(X_test)
mean_squared_error(y_test, gbr_pred), r2_score(y_test, gbr_pred)
```

    (0.21235784885823164, 0.6705746637944561)

```python
_, ax = plt.subplots()
ax.plot(alphas, gbr.cv_results_["mean_test_score"])
ax.axvline(gbr.best_params_["learning_rate"], linestyle='--'), gbr.best_params_["learning_rate"]
```

    (<matplotlib.lines.Line2D at 0x7f6cfe04e6c0>, 0.005994842503189409)
    
![png](Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_files/Tree%2C%20Bagging%2C%20Boosting%20with%20Hyperparameter%20tuning_28_1.png)

```python
caravan = load_data("Caravan")
caravan.head()
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
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>

```python
X = caravan.drop(columns=["Purchase"])
y = caravan["Purchase"]

n_train = 1000
X_train = X[:n_train]
X_test = X[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]
```

```python
gbr = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01)
gbr.fit(X_train, y_train)
```

<style>#sk-container-id-13 {
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

#sk-container-id-13 {
  color: var(--sklearn-color-text);
}

#sk-container-id-13 pre {
  padding: 0;
}

#sk-container-id-13 input.sk-hidden--visually {
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

#sk-container-id-13 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-13 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-13 div.sk-text-repr-fallback {
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

#sk-container-id-13 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-13 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-13 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-13 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-13 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-13 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-13 div.sk-serial {
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

#sk-container-id-13 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-13 label.sk-toggleable__label {
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

#sk-container-id-13 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-13 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-13 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-13 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-13 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-13 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-13 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-13 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-13 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-13 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-13 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-13 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-13 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-13 div.sk-label label.sk-toggleable__label,
#sk-container-id-13 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-13 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-13 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-13 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-13 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-13 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-13 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-13 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-13 div.sk-estimator.fitted:hover {
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

#sk-container-id-13 a.estimator_doc_link {
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

#sk-container-id-13 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-13 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-13 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-13" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" checked><label for="sk-estimator-id-25" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GradientBoostingClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">?<span>Documentation for GradientBoostingClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)</pre></div> </div></div></div></div>

```python
gbr_pred = np.where(gbr.predict_proba(X_test)[:,1] > 0.2, 'Yes', 'No')
confusion_matrix(y_test, gbr_pred), accuracy_score(y_test, gbr_pred)
```

    (array([[4334,  199],
            [ 251,   38]]),
     0.9066777270841975)

```python
gbr_pred = np.where(gbr.predict_proba(X_test)[:,1] > 0.5, 'Yes', 'No')
confusion_matrix(y_test, gbr_pred), accuracy_score(y_test, gbr_pred)
```

    (array([[4489,   44],
            [ 275,   14]]),
     0.933844877644131)

```python
38/(199+38),14/(14+44)
```

    (0.16033755274261605, 0.2413793103448276)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = np.where(lr.predict_proba(X_test)[:,1] > 0.2, 'Yes', 'No')
confusion_matrix(y_test, lr_pred), accuracy_score(y_test, lr_pred)
```

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(

    (array([[4305,  228],
            [ 240,   49]]),
     0.9029448361675654)

```python
knn = skm.GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors':range(1,11)}, scoring='precision')
knn.fit(X_train, y_train)
```

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 949, in _score
        scores = scorer(estimator, X_test, y_test, **score_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 288, in __call__
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 380, in _score
        y_pred = method_caller(
                 ^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/metrics/_scorer.py", line 90, in _cached_call
        result, _ = _get_response_values(
                    ^^^^^^^^^^^^^^^^^^^^^
      File "/home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/utils/_response.py", line 207, in _get_response_values
        raise ValueError(
    ValueError: pos_label=1 is not a valid label: It should be one of ['No' 'Yes']
    
      warnings.warn(
    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_search.py:1108: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan nan nan]
      warnings.warn(

<style>#sk-container-id-15 {
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

#sk-container-id-15 {
  color: var(--sklearn-color-text);
}

#sk-container-id-15 pre {
  padding: 0;
}

#sk-container-id-15 input.sk-hidden--visually {
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

#sk-container-id-15 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-15 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-15 div.sk-text-repr-fallback {
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

#sk-container-id-15 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-15 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-15 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-15 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-15 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-15 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-15 div.sk-serial {
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

#sk-container-id-15 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-15 label.sk-toggleable__label {
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

#sk-container-id-15 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-15 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-15 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-15 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-15 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-15 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-15 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-15 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-15 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-15 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-15 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-15 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-15 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-15 div.sk-label label.sk-toggleable__label,
#sk-container-id-15 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-15 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-15 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-15 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-15 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-15 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-15 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-15 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-15 div.sk-estimator.fitted:hover {
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

#sk-container-id-15 a.estimator_doc_link {
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

#sk-container-id-15 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-15 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-15 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-15" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(estimator=KNeighborsClassifier(),
             param_grid={&#x27;n_neighbors&#x27;: range(1, 11)}, scoring=&#x27;precision&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(estimator=KNeighborsClassifier(),
             param_grid={&#x27;n_neighbors&#x27;: range(1, 11)}, scoring=&#x27;precision&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: KNeighborsClassifier</div></div></label><div class="sk-toggleable__content fitted"><pre>KNeighborsClassifier(n_neighbors=1)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>KNeighborsClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">?<span>Documentation for KNeighborsClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>KNeighborsClassifier(n_neighbors=1)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
knn_pred = knn.best_estimator_.predict(X_test)
confusion_matrix(y_test, knn_pred), accuracy_score(y_test, knn_pred)
```

    (array([[4262,  271],
            [ 262,   27]]),
     0.8894649523019494)

```python
49/(228+49), 27/(271+27)
```

    (0.17689530685920576, 0.09060402684563758)

```python

```

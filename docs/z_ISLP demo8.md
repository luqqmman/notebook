```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from statsmodels.datasets import get_rdataset
from ISLP import load_data , confusion_table
from ISLP.models import ModelSpec as MS
from ISLP.bart import BART
```

```python
Carseats = load_data('Carseats')
Carseats
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
    </tr>
    <tr>
      <th>395</th>
      <td>12.57</td>
      <td>138</td>
      <td>108</td>
      <td>17</td>
      <td>203</td>
      <td>128</td>
      <td>Good</td>
      <td>33</td>
      <td>14</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>396</th>
      <td>6.14</td>
      <td>139</td>
      <td>23</td>
      <td>3</td>
      <td>37</td>
      <td>120</td>
      <td>Medium</td>
      <td>55</td>
      <td>11</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>397</th>
      <td>7.41</td>
      <td>162</td>
      <td>26</td>
      <td>12</td>
      <td>368</td>
      <td>159</td>
      <td>Medium</td>
      <td>40</td>
      <td>18</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>398</th>
      <td>5.94</td>
      <td>100</td>
      <td>79</td>
      <td>7</td>
      <td>284</td>
      <td>95</td>
      <td>Bad</td>
      <td>50</td>
      <td>12</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>399</th>
      <td>9.71</td>
      <td>134</td>
      <td>37</td>
      <td>0</td>
      <td>27</td>
      <td>120</td>
      <td>Good</td>
      <td>49</td>
      <td>16</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 11 columns</p>
</div>

```python
X = Carseats.drop(columns=["Sales"])
X = pd.get_dummies(X, columns=["ShelveLoc", "Urban", "US"], drop_first=True)
high_sales = Carseats.Sales > 8
```

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, high_sales, random_state=0)
```

```python
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
accuracy_score(y_test, dtc_pred)
```

    0.74

```python
ax = plt.subplots()[1]
plot_tree(dtc, feature_names=X.columns, ax=ax)
```

    [Text(0.4583333333333333, 0.875, 'Price <= 92.5\nentropy = 0.969\nsamples = 300\nvalue = [181.0, 119.0]'),
     Text(0.25, 0.625, 'ShelveLoc_Good <= 0.5\nentropy = 0.722\nsamples = 45\nvalue = [9, 36]'),
     Text(0.35416666666666663, 0.75, 'True  '),
     Text(0.16666666666666666, 0.375, 'Income <= 56.0\nentropy = 0.845\nsamples = 33\nvalue = [9, 24]'),
     Text(0.08333333333333333, 0.125, 'entropy = 0.954\nsamples = 8\nvalue = [5, 3]'),
     Text(0.25, 0.125, 'entropy = 0.634\nsamples = 25\nvalue = [4, 21]'),
     Text(0.3333333333333333, 0.375, 'entropy = 0.0\nsamples = 12\nvalue = [0, 12]'),
     Text(0.6666666666666666, 0.625, 'Advertising <= 6.5\nentropy = 0.91\nsamples = 255\nvalue = [172, 83]'),
     Text(0.5625, 0.75, '  False'),
     Text(0.5, 0.375, 'CompPrice <= 144.5\nentropy = 0.678\nsamples = 134\nvalue = [110, 24]'),
     Text(0.4166666666666667, 0.125, 'entropy = 0.534\nsamples = 115\nvalue = [101, 14]'),
     Text(0.5833333333333334, 0.125, 'entropy = 0.998\nsamples = 19\nvalue = [9, 10]'),
     Text(0.8333333333333334, 0.375, 'ShelveLoc_Good <= 0.5\nentropy = 1.0\nsamples = 121\nvalue = [62.0, 59.0]'),
     Text(0.75, 0.125, 'entropy = 0.952\nsamples = 94\nvalue = [59, 35]'),
     Text(0.9166666666666666, 0.125, 'entropy = 0.503\nsamples = 27\nvalue = [3, 24]')]
    
![png](z_ISLP%20demo8_files/z_ISLP%20demo8_5_1.png)

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, high_sales, test_size=0.5, random_state=0)
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
```

<style>#sk-container-id-2 {
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

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
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

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
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

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
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

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
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

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
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

#sk-container-id-2 a.estimator_doc_link {
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

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;)</pre></div> </div></div></div></div>

```python
ccp = dtc.cost_complexity_pruning_path(X_train, y_train)
params = {'ccp_alpha': ccp.ccp_alphas}
kfold = skm.KFold(n_splits=10, shuffle=True, random_state=0)
model = skm.GridSearchCV(dtc, param_grid=params, cv=kfold)
model.fit(X_train, y_train)
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
             estimator=DecisionTreeClassifier(criterion=&#x27;entropy&#x27;),
             param_grid={&#x27;ccp_alpha&#x27;: array([0.        , 0.01622556, 0.0171946 , 0.0180482 , 0.0180482 ,
       0.01991688, 0.02012073, 0.02070855, 0.02193427, 0.0219518 ,
       0.02220877, 0.02274806, 0.02417233, 0.02588672, 0.02714959,
       0.02735525, 0.02900052, 0.02906078, 0.03209543, 0.04499252,
       0.06236632, 0.10024835])})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=DecisionTreeClassifier(criterion=&#x27;entropy&#x27;),
             param_grid={&#x27;ccp_alpha&#x27;: array([0.        , 0.01622556, 0.0171946 , 0.0180482 , 0.0180482 ,
       0.01991688, 0.02012073, 0.02070855, 0.02193427, 0.0219518 ,
       0.02220877, 0.02274806, 0.02417233, 0.02588672, 0.02714959,
       0.02735525, 0.02900052, 0.02906078, 0.03209543, 0.04499252,
       0.06236632, 0.10024835])})</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: DecisionTreeClassifier</div></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(ccp_alpha=0.018048202372184057, criterion=&#x27;entropy&#x27;)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(ccp_alpha=0.018048202372184057, criterion=&#x27;entropy&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
model.best_estimator_.get_n_leaves(), dtc.get_n_leaves()
model.cv_results_
```

    {'mean_fit_time': array([0.00433464, 0.00290234, 0.00236988, 0.00263145, 0.00240963,
            0.00240662, 0.00256596, 0.00232456, 0.00255721, 0.00245774,
            0.00244524, 0.00223739, 0.00230029, 0.0025923 , 0.00253756,
            0.00245504, 0.00255251, 0.00247173, 0.0027205 , 0.00255225,
            0.00256662, 0.00268097]),
     'std_fit_time': array([0.00297806, 0.00059483, 0.00034463, 0.00059222, 0.00040597,
            0.00036248, 0.00050071, 0.00021018, 0.00043207, 0.00046749,
            0.0004039 , 0.00018706, 0.00025085, 0.00050736, 0.00041083,
            0.00048615, 0.00056722, 0.00048864, 0.00056649, 0.00058765,
            0.00053271, 0.00049404]),
     'mean_score_time': array([0.0029093 , 0.00165648, 0.00122352, 0.00149612, 0.00130613,
            0.00134208, 0.00132575, 0.00129905, 0.00140755, 0.00133898,
            0.00135987, 0.00122116, 0.00148056, 0.00144806, 0.00151777,
            0.00139794, 0.00143487, 0.00138805, 0.00145013, 0.00146112,
            0.0013844 , 0.00153246]),
     'std_score_time': array([0.00257493, 0.00037032, 0.00013903, 0.0004048 , 0.00034704,
            0.00033574, 0.00020024, 0.00019787, 0.0002398 , 0.00033398,
            0.00035035, 0.00018485, 0.00036048, 0.00055823, 0.00056115,
            0.00039527, 0.0003866 , 0.00035872, 0.00040886, 0.00039925,
            0.00038439, 0.00036823]),
     'param_ccp_alpha': masked_array(data=[0.0, 0.016225562489182655, 0.017194601396443954,
                        0.018048202372184057, 0.018048202372184057,
                        0.01991687712104144, 0.020120733983535546,
                        0.020708547250381463, 0.021934274536246717,
                        0.021951797627815944, 0.022208773417819985,
                        0.022748055645587552, 0.024172334280683237,
                        0.025886720366496196, 0.02714958965089037,
                        0.027355254741740692, 0.029000519903033605,
                        0.029060777786737286, 0.03209542720150803,
                        0.04499251649788086, 0.062366324849087884,
                        0.10024834625148749],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False],
            fill_value=1e+20),
     'params': [{'ccp_alpha': 0.0},
      {'ccp_alpha': 0.016225562489182655},
      {'ccp_alpha': 0.017194601396443954},
      {'ccp_alpha': 0.018048202372184057},
      {'ccp_alpha': 0.018048202372184057},
      {'ccp_alpha': 0.01991687712104144},
      {'ccp_alpha': 0.020120733983535546},
      {'ccp_alpha': 0.020708547250381463},
      {'ccp_alpha': 0.021934274536246717},
      {'ccp_alpha': 0.021951797627815944},
      {'ccp_alpha': 0.022208773417819985},
      {'ccp_alpha': 0.022748055645587552},
      {'ccp_alpha': 0.024172334280683237},
      {'ccp_alpha': 0.025886720366496196},
      {'ccp_alpha': 0.02714958965089037},
      {'ccp_alpha': 0.027355254741740692},
      {'ccp_alpha': 0.029000519903033605},
      {'ccp_alpha': 0.029060777786737286},
      {'ccp_alpha': 0.03209542720150803},
      {'ccp_alpha': 0.04499251649788086},
      {'ccp_alpha': 0.062366324849087884},
      {'ccp_alpha': 0.10024834625148749}],
     'split0_test_score': array([0.7 , 0.7 , 0.75, 0.7 , 0.75, 0.75, 0.7 , 0.7 , 0.7 , 0.7 , 0.7 ,
            0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.7 , 0.7 , 0.7 , 0.65]),
     'split1_test_score': array([0.75, 0.7 , 0.7 , 0.8 , 0.8 , 0.75, 0.85, 0.75, 0.8 , 0.75, 0.75,
            0.8 , 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.7 , 0.85, 0.75, 0.85]),
     'split2_test_score': array([0.75, 0.75, 0.7 , 0.65, 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.7 ,
            0.7 , 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.9 , 0.8 ]),
     'split3_test_score': array([0.6 , 0.65, 0.6 , 0.8 , 0.8 , 0.75, 0.75, 0.75, 0.75, 0.7 , 0.75,
            0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.6 , 0.6 , 0.45]),
     'split4_test_score': array([0.6 , 0.55, 0.6 , 0.65, 0.6 , 0.55, 0.6 , 0.6 , 0.6 , 0.6 , 0.6 ,
            0.6 , 0.6 , 0.6 , 0.6 , 0.6 , 0.6 , 0.6 , 0.6 , 0.55, 0.55, 0.55]),
     'split5_test_score': array([0.7 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 ,
            0.8 , 0.8 , 0.8 , 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.7 , 0.5 ]),
     'split6_test_score': array([0.75, 0.65, 0.6 , 0.65, 0.65, 0.65, 0.65, 0.65, 0.6 , 0.6 , 0.6 ,
            0.6 , 0.6 , 0.6 , 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.6 , 0.55]),
     'split7_test_score': array([0.65, 0.75, 0.7 , 0.75, 0.75, 0.7 , 0.75, 0.7 , 0.7 , 0.7 , 0.7 ,
            0.75, 0.65, 0.7 , 0.65, 0.7 , 0.65, 0.65, 0.7 , 0.75, 0.7 , 0.7 ]),
     'split8_test_score': array([0.7 , 0.7 , 0.75, 0.7 , 0.7 , 0.7 , 0.7 , 0.75, 0.75, 0.75, 0.75,
            0.75, 0.75, 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.6 , 0.6 , 0.45]),
     'split9_test_score': array([0.75, 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.75, 0.75, 0.75,
            0.75, 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.7 , 0.65, 0.65, 0.6 ]),
     'mean_test_score': array([0.695, 0.705, 0.7  , 0.73 , 0.735, 0.715, 0.73 , 0.72 , 0.715,
            0.705, 0.71 , 0.725, 0.71 , 0.71 , 0.695, 0.7  , 0.695, 0.695,
            0.68 , 0.675, 0.675, 0.61 ]),
     'std_test_score': array([0.05678908, 0.07228416, 0.07416198, 0.06403124, 0.06726812,
            0.07088723, 0.07141428, 0.06      , 0.06726812, 0.06103278,
            0.06244998, 0.06800735, 0.08      , 0.07681146, 0.0820061 ,
            0.08062258, 0.0820061 , 0.0820061 , 0.06      , 0.09552487,
            0.09552487, 0.13190906]),
     'rank_test_score': array([15, 11, 13,  2,  1,  6,  3,  5,  6, 12,  9,  4,  8,  9, 15, 13, 15,
            15, 19, 20, 20, 22], dtype=int32)}

```python
accuracy_score(y_test, model.best_estimator_.predict(X_test))
```

    0.705

```python
DecisionTreeClassifier?
```

    [0;31mInit signature:[0m
    [0mDecisionTreeClassifier[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0;34m*[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcriterion[0m[0;34m=[0m[0;34m'gini'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msplitter[0m[0;34m=[0m[0;34m'best'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmax_depth[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmin_samples_split[0m[0;34m=[0m[0;36m2[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmin_samples_leaf[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmin_weight_fraction_leaf[0m[0;34m=[0m[0;36m0.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmax_features[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrandom_state[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmax_leaf_nodes[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmin_impurity_decrease[0m[0;34m=[0m[0;36m0.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mclass_weight[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mccp_alpha[0m[0;34m=[0m[0;36m0.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmonotonic_cst[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    A decision tree classifier.
    
    Read more in the :ref:`User Guide <tree>`.
    
    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
    
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
    
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    
        .. versionchanged:: 0.18
           Added float values for fractions.
    
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
    
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    
        .. versionchanged:: 0.18
           Added float values for fractions.
    
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    
    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:
    
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at
          each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
    
        .. note::
    
            The search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.
    
    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
        The weighted impurity decrease equation is the following::
    
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
    
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
    
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    
        .. versionadded:: 0.19
    
    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
    
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
    
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
    
        For multi-output, the weights of each column of y will be multiplied.
    
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details. See
        :ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`
        for an example of such pruning.
    
        .. versionadded:: 0.22
    
    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease
    
        If monotonic_cst is None, no constraints are applied.
    
        Monotonicity constraints are not supported for:
          - multiclass classifications (i.e. when `n_classes > 2`),
          - multioutput classifications (i.e. when `n_outputs_ > 1`),
          - classifications trained on data with missing values.
    
        The constraints hold over the probability of the positive class.
    
        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    
        .. versionadded:: 1.4
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).
    
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
    
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    
    max_features_ : int
        The inferred value of max_features.
    
    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    
    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.
    
    See Also
    --------
    DecisionTreeRegressor : A decision tree regressor.
    
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    
    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.
    
    References
    ----------
    
    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning
    
    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    
    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    
    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    [0;31mFile:[0m           ~/Lab/islp/venv/lib/python3.12/site-packages/sklearn/tree/_classes.py
    [0;31mType:[0m           ABCMeta
    [0;31mSubclasses:[0m     ExtraTreeClassifier

```python
skm.KFold?
```

    [0;31mInit signature:[0m [0mskm[0m[0;34m.[0m[0mKFold[0m[0;34m([0m[0mn_splits[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0;34m*[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    K-Fold cross-validator.
    
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).
    
    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.
    
    Read more in the :ref:`User Guide <k_fold>`.
    
    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(kf.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2 3]
    
    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.
    
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    
    See Also
    --------
    StratifiedKFold : Takes class information into account to avoid building
        folds with imbalanced class distributions (for binary or multiclass
        classification tasks).
    
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    
    RepeatedKFold : Repeats K-Fold n times.
    [0;31mFile:[0m           ~/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_split.py
    [0;31mType:[0m           ABCMeta
    [0;31mSubclasses:[0m     

```python
skm.GridSearchCV?
```

    [0;31mInit signature:[0m
    [0mskm[0m[0;34m.[0m[0mGridSearchCV[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mestimator[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mparam_grid[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m*[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mscoring[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mn_jobs[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrefit[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcv[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mverbose[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpre_dispatch[0m[0;34m=[0m[0;34m'2*n_jobs'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0merror_score[0m[0;34m=[0m[0mnan[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mreturn_train_score[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    Exhaustive search over specified parameter values for an estimator.
    
    Important members are fit, predict.
    
    GridSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.
    
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.
    
    Read more in the :ref:`User Guide <grid_search>`.
    
    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
    
        If `scoring` represents a single score, one can use:
    
        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring_callable`) that returns a single value.
    
        If `scoring` represents multiple scores, one can use:
    
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.
    
        See :ref:`multimetric_grid_search` for an example.
    
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    
        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None
    
    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
    
        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
    
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.
    
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.
    
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
    
        See ``scoring`` parameter to know more about multiple metric
        evaluation.
    
        See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`
        to see how to design a custom selection strategy using a callable
        via `refit`.
    
        .. versionchanged:: 0.20
            Support for callable added.
    
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
    
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
    
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
    
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    
    verbose : int
        Controls the verbosity: the higher, the more messages.
    
        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.
    
    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
    
        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'
    
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    
    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
    
        .. versionadded:: 0.19
    
        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``
    
    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
    
        For instance the below given table
    
        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+
    
        will be represented by a ``cv_results_`` dict of::
    
            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }
    
        NOTE
    
        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.
    
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
    
        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)
    
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
    
        See ``refit`` parameter for more information on allowed values.
    
    best_score_ : float
        Mean cross-validated score of the best_estimator
    
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    
        This attribute is not available if ``refit`` is a function.
    
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
    
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
    
        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.
    
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
    
        This is present only if ``refit`` is not False.
    
        .. versionadded:: 0.20
    
    multimetric_ : bool
        Whether or not the scorers compute several metrics.
    
    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.
    
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.
    
        .. versionadded:: 1.0
    
    See Also
    --------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    train_test_split : Utility function to split the data into a development
        set usable for fitting a GridSearchCV instance and an evaluation set
        for its final evaluation.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.
    
    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.
    
    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    
    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split2_test_score', ...
     'std_fit_time', 'std_score_time', 'std_test_score']
    [0;31mFile:[0m           ~/Lab/islp/venv/lib/python3.12/site-packages/sklearn/model_selection/_search.py
    [0;31mType:[0m           ABCMeta
    [0;31mSubclasses:[0m     

```python
dtc.cost_complexity_pruning_path?
```

    [0;31mSignature:[0m [0mdtc[0m[0;34m.[0m[0mcost_complexity_pruning_path[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0msample_weight[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Compute the pruning path during Minimal Cost-Complexity Pruning.
    
    See :ref:`minimal_cost_complexity_pruning` for details on the pruning
    process.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples. Internally, it will be converted to
        ``dtype=np.float32`` and if a sparse matrix is provided
        to a sparse ``csc_matrix``.
    
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels) as integers or strings.
    
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted. Splits
        that would create child nodes with net zero or negative weight are
        ignored while searching for a split in each node. Splits are also
        ignored if they would result in any single class carrying a
        negative weight in either child node.
    
    Returns
    -------
    ccp_path : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
    
        ccp_alphas : ndarray
            Effective alphas of subtree during pruning.
    
        impurities : ndarray
            Sum of the impurities of the subtree leaves for the
            corresponding alpha value in ``ccp_alphas``.
    [0;31mFile:[0m      ~/Lab/islp/venv/lib/python3.12/site-packages/sklearn/tree/_classes.py
    [0;31mType:[0m      method

```python
ccp = dtc.cost_complexity_pruning_path(X_train, y_train)

```

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
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.5, random_state=0)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
```

```python
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
np.mean((y_test-dtr.predict(X_test))**2), dtr.get_n_leaves()
```

    (26.542569169960473, 238)

```python
ccp = dtr.cost_complexity_pruning_path(X_train, y_train)
params = {'ccp_alpha': ccp.ccp_alphas}
kfold = skm.KFold(n_splits=10, shuffle=True, random_state=0)
model = skm.GridSearchCV(dtr, param_grid=params, cv=kfold, scoring='neg_mean_squared_error')
model.fit(X_train, y_train)
```

<style>#sk-container-id-12 {
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

#sk-container-id-12 {
  color: var(--sklearn-color-text);
}

#sk-container-id-12 pre {
  padding: 0;
}

#sk-container-id-12 input.sk-hidden--visually {
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

#sk-container-id-12 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-12 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-12 div.sk-text-repr-fallback {
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

#sk-container-id-12 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-12 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-12 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-12 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-12 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-12 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-12 div.sk-serial {
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

#sk-container-id-12 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-12 label.sk-toggleable__label {
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

#sk-container-id-12 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-12 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-12 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-12 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-12 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-12 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-12 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-12 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-12 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-12 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-12 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-12 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-12 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-12 div.sk-label label.sk-toggleable__label,
#sk-container-id-12 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-12 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-12 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-12 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-12 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-12 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-12 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-12 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-12 div.sk-estimator.fitted:hover {
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

#sk-container-id-12 a.estimator_doc_link {
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

#sk-container-id-12 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-12 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-12 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-12" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=DecisionTreeRegressor(),
             param_grid={&#x27;ccp_alpha&#x27;: array([0.00000000e+00, 4.49355090e-16, 1.79742036e-15, 1.97628458e-05,
       1.97628458e-05, 1.97628458e-05, 1.97628458e-05, 1.97628458e-05,
       1.97628458e-05, 1.97628458e-05, 1.97628458e-05, 1.97628459e-05,
       1.97628459e-05, 1.97628459e-05, 1.97628459e-...
       1.74563061e-01, 1.84099944e-01, 1.90909989e-01, 1.98409938e-01,
       2.11783048e-01, 2.21983060e-01, 2.65387258e-01, 2.80502541e-01,
       2.95232962e-01, 4.96709706e-01, 6.35314537e-01, 6.61084017e-01,
       7.90843215e-01, 8.81586298e-01, 9.20701196e-01, 1.19620193e+00,
       2.08525213e+00, 2.91359587e+00, 5.48209786e+00, 8.19739880e+00,
       1.58561557e+01, 4.59185361e+01])},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=10, random_state=0, shuffle=True),
             estimator=DecisionTreeRegressor(),
             param_grid={&#x27;ccp_alpha&#x27;: array([0.00000000e+00, 4.49355090e-16, 1.79742036e-15, 1.97628458e-05,
       1.97628458e-05, 1.97628458e-05, 1.97628458e-05, 1.97628458e-05,
       1.97628458e-05, 1.97628458e-05, 1.97628458e-05, 1.97628459e-05,
       1.97628459e-05, 1.97628459e-05, 1.97628459e-...
       1.74563061e-01, 1.84099944e-01, 1.90909989e-01, 1.98409938e-01,
       2.11783048e-01, 2.21983060e-01, 2.65387258e-01, 2.80502541e-01,
       2.95232962e-01, 4.96709706e-01, 6.35314537e-01, 6.61084017e-01,
       7.90843215e-01, 8.81586298e-01, 9.20701196e-01, 1.19620193e+00,
       2.08525213e+00, 2.91359587e+00, 5.48209786e+00, 8.19739880e+00,
       1.58561557e+01, 4.59185361e+01])},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: DecisionTreeRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeRegressor(ccp_alpha=0.17052700922263658)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeRegressor.html">?<span>Documentation for DecisionTreeRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeRegressor(ccp_alpha=0.17052700922263658)</pre></div> </div></div></div></div></div></div></div></div></div>

```python
np.mean((y_test-model.best_estimator_.predict(X_test))**2), model.best_estimator_.get_n_leaves()
```

    (22.419123269550663, 25)

```python
_, ax = plt.subplots()
plot_tree(model.best_estimator_, feature_names=X.columns, ax=ax)
```

    [Text(0.49193548387096775, 0.9444444444444444, 'lstat <= 7.81\nsquared_error = 92.491\nsamples = 253\nvalue = 23.049'),
     Text(0.24193548387096775, 0.8333333333333334, 'rm <= 7.435\nsquared_error = 84.345\nsamples = 86\nvalue = 32.492'),
     Text(0.36693548387096775, 0.8888888888888888, 'True  '),
     Text(0.12903225806451613, 0.7222222222222222, 'dis <= 1.485\nsquared_error = 41.903\nsamples = 68\nvalue = 28.978'),
     Text(0.0967741935483871, 0.6111111111111112, 'squared_error = 0.0\nsamples = 3\nvalue = 50.0'),
     Text(0.16129032258064516, 0.6111111111111112, 'rm <= 6.556\nsquared_error = 22.499\nsamples = 65\nvalue = 28.008'),
     Text(0.12903225806451613, 0.5, 'squared_error = 2.618\nsamples = 25\nvalue = 23.748'),
     Text(0.1935483870967742, 0.5, 'rm <= 7.104\nsquared_error = 16.496\nsamples = 40\nvalue = 30.67'),
     Text(0.16129032258064516, 0.3888888888888889, 'age <= 51.1\nsquared_error = 12.905\nsamples = 30\nvalue = 29.307'),
     Text(0.0967741935483871, 0.2777777777777778, 'ptratio <= 18.85\nsquared_error = 7.328\nsamples = 18\nvalue = 30.978'),
     Text(0.06451612903225806, 0.16666666666666666, 'indus <= 6.66\nsquared_error = 5.454\nsamples = 14\nvalue = 31.9'),
     Text(0.03225806451612903, 0.05555555555555555, 'squared_error = 2.601\nsamples = 12\nvalue = 32.617'),
     Text(0.0967741935483871, 0.05555555555555555, 'squared_error = 1.0\nsamples = 2\nvalue = 27.6'),
     Text(0.12903225806451613, 0.16666666666666666, 'squared_error = 0.493\nsamples = 4\nvalue = 27.75'),
     Text(0.22580645161290322, 0.2777777777777778, 'rad <= 2.5\nsquared_error = 10.798\nsamples = 12\nvalue = 26.8'),
     Text(0.1935483870967742, 0.16666666666666666, 'squared_error = 4.562\nsamples = 5\nvalue = 24.38'),
     Text(0.25806451612903225, 0.16666666666666666, 'dis <= 3.341\nsquared_error = 8.082\nsamples = 7\nvalue = 28.529'),
     Text(0.22580645161290322, 0.05555555555555555, 'squared_error = 1.878\nsamples = 5\nvalue = 30.16'),
     Text(0.2903225806451613, 0.05555555555555555, 'squared_error = 0.302\nsamples = 2\nvalue = 24.45'),
     Text(0.22580645161290322, 0.3888888888888889, 'squared_error = 4.964\nsamples = 10\nvalue = 34.76'),
     Text(0.3548387096774194, 0.7222222222222222, 'ptratio <= 15.4\nsquared_error = 21.812\nsamples = 18\nvalue = 45.767'),
     Text(0.2903225806451613, 0.6111111111111112, 'nox <= 0.416\nsquared_error = 5.79\nsamples = 11\nvalue = 48.636'),
     Text(0.25806451612903225, 0.5, 'squared_error = 0.0\nsamples = 1\nvalue = 42.3'),
     Text(0.3225806451612903, 0.5, 'squared_error = 1.952\nsamples = 10\nvalue = 49.27'),
     Text(0.41935483870967744, 0.6111111111111112, 'age <= 44.35\nsquared_error = 13.714\nsamples = 7\nvalue = 41.257'),
     Text(0.3870967741935484, 0.5, 'squared_error = 1.749\nsamples = 3\nvalue = 44.833'),
     Text(0.45161290322580644, 0.5, 'squared_error = 5.902\nsamples = 4\nvalue = 38.575'),
     Text(0.7419354838709677, 0.8333333333333334, 'lstat <= 15.0\nsquared_error = 27.121\nsamples = 167\nvalue = 18.186'),
     Text(0.6169354838709677, 0.8888888888888888, '  False'),
     Text(0.6129032258064516, 0.7222222222222222, 'rm <= 6.53\nsquared_error = 12.255\nsamples = 87\nvalue = 21.566'),
     Text(0.5483870967741935, 0.6111111111111112, 'ptratio <= 18.65\nsquared_error = 6.248\nsamples = 74\nvalue = 20.784'),
     Text(0.5161290322580645, 0.5, 'tax <= 208.0\nsquared_error = 6.745\nsamples = 31\nvalue = 21.855'),
     Text(0.4838709677419355, 0.3888888888888889, 'squared_error = 2.56\nsamples = 2\nvalue = 28.0'),
     Text(0.5483870967741935, 0.3888888888888889, 'squared_error = 4.25\nsamples = 29\nvalue = 21.431'),
     Text(0.5806451612903226, 0.5, 'squared_error = 4.466\nsamples = 43\nvalue = 20.012'),
     Text(0.6774193548387096, 0.6111111111111112, 'ptratio <= 19.3\nsquared_error = 23.169\nsamples = 13\nvalue = 26.015'),
     Text(0.6451612903225806, 0.5, 'rm <= 6.947\nsquared_error = 10.304\nsamples = 10\nvalue = 27.98'),
     Text(0.6129032258064516, 0.3888888888888889, 'squared_error = 5.648\nsamples = 7\nvalue = 26.429'),
     Text(0.6774193548387096, 0.3888888888888889, 'squared_error = 2.447\nsamples = 3\nvalue = 31.6'),
     Text(0.7096774193548387, 0.5, 'squared_error = 10.302\nsamples = 3\nvalue = 19.467'),
     Text(0.8709677419354839, 0.7222222222222222, 'dis <= 1.918\nsquared_error = 17.363\nsamples = 80\nvalue = 14.511'),
     Text(0.8064516129032258, 0.6111111111111112, 'tax <= 551.5\nsquared_error = 13.188\nsamples = 36\nvalue = 11.672'),
     Text(0.7741935483870968, 0.5, 'squared_error = 3.129\nsamples = 9\nvalue = 15.756'),
     Text(0.8387096774193549, 0.5, 'lstat <= 19.645\nsquared_error = 9.131\nsamples = 27\nvalue = 10.311'),
     Text(0.8064516129032258, 0.3888888888888889, 'squared_error = 4.756\nsamples = 5\nvalue = 13.8'),
     Text(0.8709677419354839, 0.3888888888888889, 'crim <= 30.201\nsquared_error = 6.73\nsamples = 22\nvalue = 9.518'),
     Text(0.8387096774193549, 0.2777777777777778, 'squared_error = 5.059\nsamples = 18\nvalue = 10.217'),
     Text(0.9032258064516129, 0.2777777777777778, 'squared_error = 2.172\nsamples = 4\nvalue = 6.375'),
     Text(0.9354838709677419, 0.6111111111111112, 'crim <= 0.593\nsquared_error = 8.79\nsamples = 44\nvalue = 16.834'),
     Text(0.9032258064516129, 0.5, 'squared_error = 5.269\nsamples = 16\nvalue = 19.363'),
     Text(0.967741935483871, 0.5, 'squared_error = 5.061\nsamples = 28\nvalue = 15.389')]
    
![png](z_ISLP%20demo8_files/z_ISLP%20demo8_20_1.png)

```python
bag = RandomForestRegressor(max_features=X_train.shape[1], random_state=0)
bag.fit(X_train, y_train)
np.mean((y_test - bag.predict(X_test))**2)
```

    15.998719418972334

```python
rf = RandomForestRegressor(max_features="sqrt", random_state=0)
rf.fit(X_train, y_train)
np.mean((y_test - rf.predict(X_test))**2)
```

    15.54812255731225

```python
pd.DataFrame({"importance": rf.feature_importances_}, index=X_train.columns).sort_values(by="importance")

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
      <th>chas</th>
      <td>0.008253</td>
    </tr>
    <tr>
      <th>rad</th>
      <td>0.010506</td>
    </tr>
    <tr>
      <th>zn</th>
      <td>0.017884</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.034596</td>
    </tr>
    <tr>
      <th>tax</th>
      <td>0.040578</td>
    </tr>
    <tr>
      <th>indus</th>
      <td>0.048895</td>
    </tr>
    <tr>
      <th>dis</th>
      <td>0.055616</td>
    </tr>
    <tr>
      <th>crim</th>
      <td>0.068045</td>
    </tr>
    <tr>
      <th>nox</th>
      <td>0.073336</td>
    </tr>
    <tr>
      <th>ptratio</th>
      <td>0.076813</td>
    </tr>
    <tr>
      <th>rm</th>
      <td>0.275036</td>
    </tr>
    <tr>
      <th>lstat</th>
      <td>0.290442</td>
    </tr>
  </tbody>
</table>
</div>

```python
gbr = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.01, random_state=0)
gbr.fit(X_train, y_train)
np.mean((y_test-gbr.predict(X_test))**2)
```

    15.118799219930267

```python
mse = np.zeros_like(gbr.train_score_)
for i, y_ in enumerate(gbr.staged_predict(X_test)):
    mse[i] = np.mean((y_test-y_)**2)

_, ax = plt.subplots()
ax.plot(np.arange(len(mse)), gbr.train_score_, 'b', label="train")
ax.plot(np.arange(len(mse)), mse, 'r', label="test")
ax.legend()
```

    <matplotlib.legend.Legend at 0x7eff603da810>
    
![png](z_ISLP%20demo8_files/z_ISLP%20demo8_25_1.png)

```python
pd.DataFrame({"importance": gbr.feature_importances_}, index=X_train.columns).sort_values(by="importance")
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
      <th>zn</th>
      <td>0.000377</td>
    </tr>
    <tr>
      <th>chas</th>
      <td>0.001123</td>
    </tr>
    <tr>
      <th>rad</th>
      <td>0.002500</td>
    </tr>
    <tr>
      <th>indus</th>
      <td>0.004539</td>
    </tr>
    <tr>
      <th>nox</th>
      <td>0.009962</td>
    </tr>
    <tr>
      <th>tax</th>
      <td>0.012848</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.013178</td>
    </tr>
    <tr>
      <th>crim</th>
      <td>0.029443</td>
    </tr>
    <tr>
      <th>ptratio</th>
      <td>0.044015</td>
    </tr>
    <tr>
      <th>dis</th>
      <td>0.062999</td>
    </tr>
    <tr>
      <th>rm</th>
      <td>0.278958</td>
    </tr>
    <tr>
      <th>lstat</th>
      <td>0.540057</td>
    </tr>
  </tbody>
</table>
</div>

```python
gbr?
```

    Object `gbr` not found.

```python

```

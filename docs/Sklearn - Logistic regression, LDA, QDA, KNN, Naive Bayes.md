13. This question should be answered using the Weekly data set, which
is part of the ISLP package. This data is similar in nature to the
Smarket data from this chapter’s lab, except that it contains 1, 089
weekly returns for 21 years, from the beginning of 1990 to the end of
2010.
(a) Produce some numerical and graphical summaries of the Weekly
data. Do there appear to be any patterns?
(b) Use the full data set to perform a logistic regression with
Direction as the response and the five lag variables plus Volume
as predictors. Use the summary function to print the results. Do
any of the predictors appear to be statistically significant? If so,
which ones?
(c) Compute the confusion matrix and overall fraction of correct
predictions. Explain what the confusion matrix is telling you
about the types of mistakes made by logistic regression.
(d) Now fit the logistic regression model using a training data period
from 1990 to 2008, with Lag2 as the only predictor. Compute the
confusion matrix and the overall fraction of correct predictions
for the held out data (that is, the data from 2009 and 2010).
(e) Repeat (d) using LDA.
(f) Repeat (d) using QDA.
(g) Repeat (d) using KNN with K = 1.
(h) Repeat (d) using naive Bayes.
(i) Which of these methods appears to provide the best results on
this data?
(j) Experiment with different combinations of predictors, includ-
ing possible transformations and interactions, for each of the
methods. Report the variables, method, and associated confu-
sion matrix that appears to provide the best results on the held
out data. Note that you should also experiment with values for
K in the KNN classifier.

```python
from ISLP import load_data
```

```python
weekly = load_data("Weekly")
weekly.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1089 entries, 0 to 1088
    Data columns (total 9 columns):
     #   Column     Non-Null Count  Dtype   
    ---  ------     --------------  -----   
     0   Year       1089 non-null   int64   
     1   Lag1       1089 non-null   float64 
     2   Lag2       1089 non-null   float64 
     3   Lag3       1089 non-null   float64 
     4   Lag4       1089 non-null   float64 
     5   Lag5       1089 non-null   float64 
     6   Volume     1089 non-null   float64 
     7   Today      1089 non-null   float64 
     8   Direction  1089 non-null   category
    dtypes: category(1), float64(7), int64(1)
    memory usage: 69.4 KB

```python
weekly.describe()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2000.048669</td>
      <td>0.150585</td>
      <td>0.151079</td>
      <td>0.147205</td>
      <td>0.145818</td>
      <td>0.139893</td>
      <td>1.574618</td>
      <td>0.149899</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.033182</td>
      <td>2.357013</td>
      <td>2.357254</td>
      <td>2.360502</td>
      <td>2.360279</td>
      <td>2.361285</td>
      <td>1.686636</td>
      <td>2.356927</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>0.087465</td>
      <td>-18.195000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1995.000000</td>
      <td>-1.154000</td>
      <td>-1.154000</td>
      <td>-1.158000</td>
      <td>-1.158000</td>
      <td>-1.166000</td>
      <td>0.332022</td>
      <td>-1.154000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2000.000000</td>
      <td>0.241000</td>
      <td>0.241000</td>
      <td>0.241000</td>
      <td>0.238000</td>
      <td>0.234000</td>
      <td>1.002680</td>
      <td>0.241000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2005.000000</td>
      <td>1.405000</td>
      <td>1.409000</td>
      <td>1.409000</td>
      <td>1.409000</td>
      <td>1.405000</td>
      <td>2.053727</td>
      <td>1.405000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2010.000000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>9.328214</td>
      <td>12.026000</td>
    </tr>
  </tbody>
</table>
</div>

```python
import seaborn as sns
sns.pairplot(weekly, hue="Direction")
```

    <seaborn.axisgrid.PairGrid at 0x7224845a0d40>
    
![png](Sklearn%20-%20Logistic%20regression%2C%20LDA%2C%20QDA%2C%20KNN%2C%20Naive%20Bayes_files/Sklearn%20-%20Logistic%20regression%2C%20LDA%2C%20QDA%2C%20KNN%2C%20Naive%20Bayes_4_1.png)

```python
weekly_x = weekly.drop(columns=["Direction", "Year", "Today"])
weekly_y = weekly["Direction"]
```

```python
weekly_x.corr()
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
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lag1</th>
      <td>1.000000</td>
      <td>-0.074853</td>
      <td>0.058636</td>
      <td>-0.071274</td>
      <td>-0.008183</td>
      <td>-0.064951</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>-0.074853</td>
      <td>1.000000</td>
      <td>-0.075721</td>
      <td>0.058382</td>
      <td>-0.072499</td>
      <td>-0.085513</td>
    </tr>
    <tr>
      <th>Lag3</th>
      <td>0.058636</td>
      <td>-0.075721</td>
      <td>1.000000</td>
      <td>-0.075396</td>
      <td>0.060657</td>
      <td>-0.069288</td>
    </tr>
    <tr>
      <th>Lag4</th>
      <td>-0.071274</td>
      <td>0.058382</td>
      <td>-0.075396</td>
      <td>1.000000</td>
      <td>-0.075675</td>
      <td>-0.061075</td>
    </tr>
    <tr>
      <th>Lag5</th>
      <td>-0.008183</td>
      <td>-0.072499</td>
      <td>0.060657</td>
      <td>-0.075675</td>
      <td>1.000000</td>
      <td>-0.058517</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>-0.064951</td>
      <td>-0.085513</td>
      <td>-0.069288</td>
      <td>-0.061075</td>
      <td>-0.058517</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.linear_model import LogisticRegression
from ISLP import confusion_table
```

```python
lr = LogisticRegression(C=10e10, solver='liblinear')
lr.fit(weekly_x, weekly_y)
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
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=100000000000.0, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LogisticRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(C=100000000000.0, solver=&#x27;liblinear&#x27;)</pre></div> </div></div></div></div>

```python
lr.coef_
```

    array([[-0.04126692,  0.05843996, -0.01606072, -0.02778961, -0.01447045,
            -0.02274015]])

```python
lr_pred = lr.predict(weekly_x)
confusion_table(lr_pred, weekly_y)
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
      <th>Truth</th>
      <th>Down</th>
      <th>Up</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>54</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>430</td>
      <td>557</td>
    </tr>
  </tbody>
</table>
</div>

```python
(lr_pred == weekly_y).mean()
```

    0.5610651974288338

```python
train = weekly["Year"] <= 2008
```

```python
X_train = (weekly[["Lag2"]])[train]
y_train = (weekly["Direction"])[train]
X_test = (weekly[["Lag2"]])[~train]
y_test = (weekly["Direction"])[~train]
```

```python
lr = LogisticRegression(C=10e10, solver='liblinear')
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
confusion_table(lr_pred, y_test)
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
      <th>Truth</th>
      <th>Down</th>
      <th>Up</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>34</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```

```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
confusion_table(lda_pred, y_test)
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
      <th>Truth</th>
      <th>Down</th>
      <th>Up</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>34</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>

```python
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
confusion_table(qda_pred, y_test)
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
      <th>Truth</th>
      <th>Down</th>
      <th>Up</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>43</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>

```python
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
confusion_table(nb_pred, y_test)
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
      <th>Truth</th>
      <th>Down</th>
      <th>Up</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>43</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>

```python
for i in range(1, 10):
    print(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print(confusion_table(knn_pred, y_test))
    print()
```

    1
    Truth      Down  Up
    Predicted          
    Down         22  32
    Up           21  29
    
    2
    Truth      Down  Up
    Predicted          
    Down         31  44
    Up           12  17
    
    3
    Truth      Down  Up
    Predicted          
    Down         16  19
    Up           27  42
    
    4
    Truth      Down  Up
    Predicted          
    Down         26  27
    Up           17  34
    
    5
    Truth      Down  Up
    Predicted          
    Down         16  21
    Up           27  40
    
    6
    Truth      Down  Up
    Predicted          
    Down         20  28
    Up           23  33
    
    7
    Truth      Down  Up
    Predicted          
    Down         16  20
    Up           27  41
    
    8
    Truth      Down  Up
    Predicted          
    Down         21  25
    Up           22  36
    
    9
    Truth      Down  Up
    Predicted          
    Down         17  20
    Up           26  41

```python
weekly[["Today", "Direction"]]
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
      <th>Today</th>
      <th>Direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.270</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.576</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.514</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.712</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.178</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>2.969</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>1085</th>
      <td>1.281</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>1086</th>
      <td>0.283</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>1.034</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>0.069</td>
      <td>Up</td>
    </tr>
  </tbody>
</table>
<p>1089 rows × 2 columns</p>
</div>

```python

```

# Deep Learning but Linear Regression, Logistic Regression, and SVM
This time we will:
- build simple Linear Regression, Logistic Regression, and SVM but using neural network
- comparing the result from sklearn library and our own neural network version of those models
- add hidden layer as 'feature extraction' to see any improvement (and justify the title xd)

```python
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
from ISLP import load_data
```

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, RMSprop, SGD
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchinfo import summary

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
```

```python
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
```

# Dataset for regression and classification
We will be using hitters dataset from ISLP library which about baseball player game statistic:
- regression: predict salary based on player statistic
- classification: predict if salary is above or below median

Preprocessing:
- remove null value
- handle categorical data with dummy variable
- it IMPORTANT to drop the first category because of perfect multicolinearity
- standarization

```python
hitters = load_data('Hitters')
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
hitters = pd.get_dummies(hitters, columns=['League', 'Division', 'NewLeague'], drop_first=True)
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
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>Salary</th>
      <th>League_N</th>
      <th>Division_W</th>
      <th>NewLeague_N</th>
    </tr>
  </thead>
  <tbody>
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
      <td>632</td>
      <td>43</td>
      <td>10</td>
      <td>475.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
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
      <td>880</td>
      <td>82</td>
      <td>14</td>
      <td>480.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
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
      <td>200</td>
      <td>11</td>
      <td>3</td>
      <td>500.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
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
      <td>805</td>
      <td>40</td>
      <td>4</td>
      <td>91.5</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>594</td>
      <td>169</td>
      <td>4</td>
      <td>74</td>
      <td>51</td>
      <td>35</td>
      <td>11</td>
      <td>4408</td>
      <td>1133</td>
      <td>19</td>
      <td>501</td>
      <td>336</td>
      <td>194</td>
      <td>282</td>
      <td>421</td>
      <td>25</td>
      <td>750.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

```python
X_reg = hitters.drop(columns=["Salary"])
y_reg = hitters["Salary"]
X_train, X_test, y_train, y_test = skm.train_test_split(X_reg, y_reg, shuffle=True, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Generic Model for our task
We will use this model for our task, this is very straightforward model with only input layer and one output unit. The only main thing that we need to think about is the different loss function for Linear Regression, Logistic Regression, and SVM. There is also simple hidden layer with 32 unit that we will compare the performance later.

```python
class Model(pl.LightningModule):
    def __init__(self, loss_fn, input_size, hidden_layer=False, lr=0.01):
        super().__init__()
        self.loss_fn = loss_fn
        self.input_size = input_size
        self.lr = lr
        if hidden_layer:
            self.model = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, 1)
            )

    def forward(self, x):
        return self.model(x).squeeze()

    def _shared_step(self, batch, stage):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y.float())
        self.log(f'{stage}_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

class DataModule(pl.LightningDataModule):
    def __init__(self, train_td, val_td, batch_size=64, num_workers=8):
        super().__init__()
        self.train_td = train_td
        self.val_td = val_td
        self.batch_size = batch_size
        self.num_workers=num_workers

    def train_dataloader(self):
        return DataLoader(self.train_td, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_td, batch_size=self.batch_size, num_workers=self.num_workers)    
```

## Linear Regression with sklearn
This is the default linear regression model from sklearn. It got MAE about 265 and 0.55 R2 score. For reference the std of our response is around 451 so it did pretty good with MAE almost half of std.

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
mean_absolute_error(y_test, lr_pred), r2_score(y_test, lr_pred)
```

    (265.87464986111604, 0.5531784057871476)

## Linear Regression with our NN model
This is the our NN model which use MSELoss. The result is bad here, the validation and train loss get stuck at some point which adding hidden layer maybe helpful.

```python
X_train_t = torch.tensor(X_train.astype(np.float32))
X_test_t = torch.tensor(X_test.astype(np.float32))
y_train_t = torch.tensor(y_train.to_numpy().astype(np.float32))
y_test_t = torch.tensor(y_test.to_numpy().astype(np.float32))

train_lr_ds = TensorDataset(X_train_t, y_train_t)
test_lr_ds = TensorDataset(X_test_t, y_test_t)
```

```python
lr_loss_fn = nn.MSELoss()
# lr_logger = CSVLogger('logs', name='linear_regression')
lr_model = Model(lr_loss_fn, X_train.shape[1])
lr_dm = DataModule(train_lr_ds, test_lr_ds, batch_size=32)
lr_trainer = Trainer(max_epochs=100)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
lr_trainer.fit(lr_model, lr_dm)
```
    
      | Name    | Type       | Params | Mode 
    -----------------------------------------------
    0 | loss_fn | MSELoss    | 0      | train
    1 | model   | Sequential | 20     | train
    -----------------------------------------------
    20        Trainable params
    0         Non-trainable params
    20        Total params
    0.000     Total estimated model params size (MB)
    3         Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (7) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

    `Trainer.fit` stopped: `max_epochs=100` reached.

```python
lr_dl_pred = np.concatenate(lr_trainer.predict(lr_model, lr_dm.val_dataloader()))
mean_absolute_error(y_test, lr_dl_pred), r2_score(y_test, lr_dl_pred)
```

    Predicting: |                                             | 0/? [00:00<?, ?it/s]

    (314.38981907700054, 0.4528384642986234)

## Not so 'Linear' Regression with hidden layer
We will try adding hidden layer here to compromise with bad result that we get previously. I learn about exploding gradient while running this, so i set the learning rate very low but with more epoch. Sometime the loss function get stuck around the same point loss value with our previous model. But this time the model escape that local minimum and get much better result than sklearn model at around 224 MAE and 0.68 R2 score.

```python
lr_loss_fn = nn.MSELoss()
lr_model = Model(lr_loss_fn, X_train.shape[1], hidden_layer=True, lr=0.00005)
lr_dm = DataModule(train_lr_ds, test_lr_ds, batch_size=64)
lr_trainer = Trainer(max_epochs=300)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
lr_trainer.fit(lr_model, lr_dm)
```
    
      | Name    | Type       | Params | Mode 
    -----------------------------------------------
    0 | loss_fn | MSELoss    | 0      | train
    1 | model   | Sequential | 673    | train
    -----------------------------------------------
    673       Trainable params
    0         Non-trainable params
    673       Total params
    0.003     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode

    `Trainer.fit` stopped: `max_epochs=300` reached.

```python
lr_dlh_pred = np.concatenate(lr_trainer.predict(lr_model, lr_dm.val_dataloader()))
mean_absolute_error(y_test, lr_dlh_pred), r2_score(y_test, lr_dlh_pred)
```

    Predicting: |                                             | 0/? [00:00<?, ?it/s]

    (224.00030996264832, 0.6806740806956346)

## Logistic Regression with sklearn
This time we will change the response to True or False, is the salary is higher or lower than median salary. The sklearn Logistic Regression model already giving a good start at around 81.8% accuracy.

```python
X_cls = hitters.drop(columns=["Salary"])
y_cls = hitters["Salary"] > hitters["Salary"].median()
X_train, X_test, y_train, y_test = skm.train_test_split(X_cls, y_cls, shuffle=True, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```python
lg = LogisticRegression().fit(X_train, y_train)
lg_pred = lg.predict(X_test)
accuracy_score(y_test, lg_pred)
```

    0.8181818181818182

## Logistic Regression with our NN model
We will try to recreate the idea of Logistic Regression model with our NN model. The key is to use this BCEWithLogitLoss which will do sigmoid for us and use binary cross entropy as loss function. We only provide the logits in output unit so to predict our test data we need to do sigmoid first to map the logit range to probability.

The result is a bit better with 83.3% accuracy

```python
X_train_t = torch.tensor(X_train.astype(np.float32), dtype=torch.float32)
X_test_t = torch.tensor(X_test.astype(np.float32), dtype=torch.float32)
y_train_t = torch.tensor(y_train.to_numpy())
y_test_t = torch.tensor(y_test.to_numpy())

train_lg_ds = TensorDataset(X_train_t, y_train_t)
test_lg_ds = TensorDataset(X_test_t, y_test_t)
```

```python
lg_loss_fn = nn.BCEWithLogitsLoss()
lg_model = Model(lg_loss_fn, X_train.shape[1], lr=0.01)
lg_dm = DataModule(train_lg_ds, test_lg_ds, batch_size=32)
lg_trainer = Trainer(max_epochs=100)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
lg_trainer.fit(lg_model, lg_dm)
```
    
      | Name    | Type              | Params | Mode 
    ------------------------------------------------------
    0 | loss_fn | BCEWithLogitsLoss | 0      | train
    1 | model   | Sequential        | 20     | train
    ------------------------------------------------------
    20        Trainable params
    0         Non-trainable params
    20        Total params
    0.000     Total estimated model params size (MB)
    3         Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (7) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

    `Trainer.fit` stopped: `max_epochs=100` reached.

```python
lg_dl_pred = np.concatenate(lg_trainer.predict(lg_model, lg_dm.val_dataloader()))
lg_dl_pred = torch.sigmoid(torch.tensor(lg_dl_pred)).numpy()
accuracy_score(y_test, np.where(lg_dl_pred>0.5, 1, 0))
```

    Predicting: |                                             | 0/? [00:00<?, ?it/s]

    0.8333333333333334

## Adding hidden layer to our Logistic NN
We will add hidden layer this time and the result is the same as before. We will try to do another method for this classification task, that is using SVM

```python
lg_loss_fn = nn.BCEWithLogitsLoss()
lg_metrics = BinaryAccuracy()
lg_logger = CSVLogger('logs', name='logistic_regression')
lg_model = Model(lg_loss_fn, X_train.shape[1], hidden_layer=True, lr=0.01)
lg_dm = DataModule(train_lg_ds, test_lg_ds, batch_size=32)
lg_trainer = Trainer(max_epochs=100)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
lg_trainer.fit(lg_model, lg_dm)
```
    
      | Name    | Type              | Params | Mode 
    ------------------------------------------------------
    0 | loss_fn | BCEWithLogitsLoss | 0      | train
    1 | model   | Sequential        | 673    | train
    ------------------------------------------------------
    673       Trainable params
    0         Non-trainable params
    673       Total params
    0.003     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (7) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

    `Trainer.fit` stopped: `max_epochs=100` reached.

```python
lg_dl_pred = np.concatenate(lg_trainer.predict(lg_model, lg_dm.val_dataloader()))
lg_dl_pred = torch.sigmoid(torch.tensor(lg_dl_pred)).numpy()
accuracy_score(y_test, np.where(lg_dl_pred>0.5, 1, 0))
```

    Predicting: |                                             | 0/? [00:00<?, ?it/s]

    0.8333333333333334

```python

```

## SVM with sklearn
Logistic regression for this classification task result around 83%. This time we will take different approach that is using support vector machine to split the feature space to two different region. The sklearn linear SVM result around 80 accuracy.

```python
svc = SVC(kernel='linear').fit(X_train, y_train)
svc_pred = svc.predict(X_test)
accuracy_score(y_test, np.where(svc_pred>0.5, 1, 0))
```

    0.803030303030303

## SVM with our NN model
We will use our NN to build SVM, but we need to define our own hinge loss as the loss function for this task. One thing that i learn is torch can do backprop even without their nn.loss function. The decision is the same with SMV, if the output is positive then it's above median and vice versa. The result without hidden layer is 81.8% accuracy.

```python
def hinge_loss(outputs, labels):
    labels = labels.float()
    labels = labels * 2 - 1
    loss = torch.mean(torch.clamp(1 - outputs.view(-1) * labels, min=0))
    return loss
```

```python
svm_loss_fn = hinge_loss
svm_model = Model(svm_loss_fn, X_train.shape[1], lr=0.01)
svm_dm = DataModule(train_lg_ds, test_lg_ds, batch_size=32)
svm_trainer = Trainer(max_epochs=100)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
svm_trainer.fit(svm_model, svm_dm)
```
    
      | Name  | Type       | Params | Mode 
    ---------------------------------------------
    0 | model | Sequential | 20     | train
    ---------------------------------------------
    20        Trainable params
    0         Non-trainable params
    20        Total params
    0.000     Total estimated model params size (MB)
    2         Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (7) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

    `Trainer.fit` stopped: `max_epochs=100` reached.

```python
svm_dl_pred = np.concatenate(svm_trainer.predict(svm_model, svm_dm.val_dataloader()))
svm_dl_pred = torch.tensor(svm_dl_pred).numpy()
accuracy_score(y_test, np.where(svm_dl_pred>0, 1, 0))
```

    Predicting: |                                             | 0/? [00:00<?, ?it/s]

    0.8181818181818182

## Adding hidden layer to our SVM model
Adding hidden layer in this case doesn't improve our model just like in logistic regression case. Overfitting may be the cause, as we dont do any regularization in our model

```python
svm_loss_fn = hinge_loss
svm_model = Model(svm_loss_fn, X_train.shape[1], lr=0.01, hidden_layer=True)
svm_dm = DataModule(train_lg_ds, test_lg_ds, batch_size=32)
svm_trainer = Trainer(max_epochs=100)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
svm_trainer.fit(svm_model, svm_dm)
```
    
      | Name  | Type       | Params | Mode 
    ---------------------------------------------
    0 | model | Sequential | 673    | train
    ---------------------------------------------
    673       Trainable params
    0         Non-trainable params
    673       Total params
    0.003     Total estimated model params size (MB)
    4         Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (7) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

    `Trainer.fit` stopped: `max_epochs=100` reached.

```python
svm_dl_pred = np.concatenate(svm_trainer.predict(svm_model, svm_dm.val_dataloader()))
svm_dl_pred = torch.tensor(svm_dl_pred).numpy()
accuracy_score(y_test, np.where(svm_dl_pred>0, 1, 0))
```

    Predicting: |                                             | 0/? [00:00<?, ?it/s]

    0.803030303030303

```python

```

```python

```

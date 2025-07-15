# NYSE trading volume estimation with AutoRegression, RNN, and fusion of LSTM + Dense
This exercise is from Introduction of statistical analysis (ISLP) Section DeepLearning.

We will predict trading volume from this three time series predictor.
- Log trading volume. This is the fraction of all outstanding shares that
are traded on that day, relative to a 100-day moving average of past
- Dow Jones return. This is the difference between the log of the Dow
Jones Industrial Index on consecutive trading days.
- Log volatility. This is based on the absolute values of daily price
movements.

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
from torch.optim import Adam, RMSprop
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
```

```python
NYSE = load_data('NYSE')
NYSE.index = pd.to_datetime(NYSE.index)
cols = ['DJ_return', 'log_volume', 'log_volatility']
X = pd.DataFrame(
    StandardScaler(with_mean=True, with_std=True).fit_transform(NYSE[cols]),
    columns=NYSE[cols].columns,
    index=NYSE.index
)
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
      <th>DJ_return</th>
      <th>log_volume</th>
      <th>log_volatility</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1962-12-03</th>
      <td>-0.549823</td>
      <td>0.175075</td>
      <td>-4.357078</td>
    </tr>
    <tr>
      <th>1962-12-04</th>
      <td>0.905200</td>
      <td>1.517291</td>
      <td>-2.529058</td>
    </tr>
    <tr>
      <th>1962-12-05</th>
      <td>0.434813</td>
      <td>2.283789</td>
      <td>-2.418037</td>
    </tr>
    <tr>
      <th>1962-12-06</th>
      <td>-0.431397</td>
      <td>0.935176</td>
      <td>-2.366521</td>
    </tr>
    <tr>
      <th>1962-12-07</th>
      <td>0.046340</td>
      <td>0.224779</td>
      <td>-2.500970</td>
    </tr>
  </tbody>
</table>
</div>

```python
## Preprocess the lag, adding day of week and month as hot one encoding
```

```python
for lag in range(1, 6):
    for col in cols:
        newcol = np.zeros(X.shape [0]) * np.nan
        newcol[lag:] = X[col]. values[:-lag]
        X.insert(len(X.columns), "{0}_{1}".format(col , lag), newcol)
X.insert(len(X.columns), 'train', NYSE['train'])
X = X.dropna()
```

```python
Y, train = X['log_volume'], X['train'].values
X = X.drop(columns =['train'] + cols)
X.columns
```

    Index(['DJ_return_1', 'log_volume_1', 'log_volatility_1', 'DJ_return_2',
           'log_volume_2', 'log_volatility_2', 'DJ_return_3', 'log_volume_3',
           'log_volatility_3', 'DJ_return_4', 'log_volume_4', 'log_volatility_4',
           'DJ_return_5', 'log_volume_5', 'log_volatility_5'],
          dtype='object')

```python
X_tmp = X.join(NYSE["day_of_week"])
X_tmp['month'] = X_tmp.index.month
X_tmp = pd.get_dummies(X_tmp, columns=["day_of_week", "month"])
X_tmp.head()
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
      <th>DJ_return_1</th>
      <th>log_volume_1</th>
      <th>log_volatility_1</th>
      <th>DJ_return_2</th>
      <th>log_volume_2</th>
      <th>log_volatility_2</th>
      <th>DJ_return_3</th>
      <th>log_volume_3</th>
      <th>log_volatility_3</th>
      <th>DJ_return_4</th>
      <th>...</th>
      <th>month_3</th>
      <th>month_4</th>
      <th>month_5</th>
      <th>month_6</th>
      <th>month_7</th>
      <th>month_8</th>
      <th>month_9</th>
      <th>month_10</th>
      <th>month_11</th>
      <th>month_12</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1962-12-10</th>
      <td>0.046340</td>
      <td>0.224779</td>
      <td>-2.500970</td>
      <td>-0.431397</td>
      <td>0.935176</td>
      <td>-2.366521</td>
      <td>0.434813</td>
      <td>2.283789</td>
      <td>-2.418037</td>
      <td>0.905200</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1962-12-11</th>
      <td>-1.304126</td>
      <td>0.605918</td>
      <td>-1.366028</td>
      <td>0.046340</td>
      <td>0.224779</td>
      <td>-2.500970</td>
      <td>-0.431397</td>
      <td>0.935176</td>
      <td>-2.366521</td>
      <td>0.434813</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1962-12-12</th>
      <td>-0.006294</td>
      <td>-0.013661</td>
      <td>-1.505667</td>
      <td>-1.304126</td>
      <td>0.605918</td>
      <td>-1.366028</td>
      <td>0.046340</td>
      <td>0.224779</td>
      <td>-2.500970</td>
      <td>-0.431397</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1962-12-13</th>
      <td>0.377081</td>
      <td>0.042552</td>
      <td>-1.551515</td>
      <td>-0.006294</td>
      <td>-0.013661</td>
      <td>-1.505667</td>
      <td>-1.304126</td>
      <td>0.605918</td>
      <td>-1.366028</td>
      <td>0.046340</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1962-12-14</th>
      <td>-0.411718</td>
      <td>-0.419836</td>
      <td>-1.597607</td>
      <td>0.377081</td>
      <td>0.042552</td>
      <td>-1.551515</td>
      <td>-0.006294</td>
      <td>-0.013661</td>
      <td>-1.505667</td>
      <td>-1.304126</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>

## Linear Regression as base measure

```python
lr = LinearRegression().fit(X[train], Y[train])
lr_pred = lr.predict(X[~train])
((lr_pred - Y[~train])**2).mean(), r2_score(Y[~train], lr_pred)
```

    (0.6186285663937958, 0.4128912938562521)

## AutoRegression
9. Fit a lag-5 autoregressive model to the NYSE data, as described in
the text and Lab 10.9.6. Refit the model with a 12-level factor repre-
senting the month. Does this factor improve the performance of the
model?

Improve R2 by around 5%

```python
datasets = []
for mask in [train, ~train]:
    X_day_month_t = torch.tensor(
        np.asarray(X_tmp[mask]).astype(np.float32)
    )
    Y_t = torch.tensor(
        np.asarray(Y[mask]).astype(np.float32)
    )
    datasets.append(TensorDataset(X_day_month_t , Y_t))
ar_train, ar_test = datasets
```

```python
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from torchmetrics import R2Score

class AutoRegression(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.learning_rate = 0.001
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(self, x):
        return self.model(x).squeeze()

    def _shared_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_r2.update(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_r2.update(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_r2.update(preds, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
```

```python
class AutoRegressionDataModule(pl.LightningDataModule):
    def __init__(self, train_td, test_td, batch_size=32, num_workers=8):
        super().__init__()
        self.train_td = train_td
        self.test_td = test_td
        self.batch_size = batch_size
        self.num_workers=num_workers

    def train_dataloader(self):
        return DataLoader(self.train_td, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return self.test_dataloader()
    
    def test_dataloader(self):
        return DataLoader(self.test_td, batch_size=self.batch_size, num_workers=self.num_workers)
```

```python
ar_model = AutoRegression(len(X_tmp.columns))
ar_dm = AutoRegressionDataModule(ar_train, ar_test)
ar_logger = TensorBoardLogger('logs', name='NYSE_AR')
ar_trainer = Trainer(
    deterministic=True, 
    max_epochs=30, 
    logger=ar_logger,
    callbacks=EarlyStopping(monitor='val_loss', patience=5, mode='min')
)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
ar_trainer.fit(ar_model, datamodule=ar_dm)
```
    
      | Name     | Type       | Params | Mode 
    ------------------------------------------------
    0 | model    | Sequential | 341    | train
    1 | loss_fn  | MSELoss    | 0      | train
    2 | train_r2 | R2Score    | 0      | train
    3 | val_r2   | R2Score    | 0      | train
    4 | test_r2  | R2Score    | 0      | train
    ------------------------------------------------
    341       Trainable params
    0         Non-trainable params
    341       Total params
    0.001     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode

```python
ar_trainer.test(ar_model, datamodule=ar_dm)
```

    Testing: |                                                | 0/? [00:00<?, ?it/s]

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_loss           0.5618957877159119
             test_r2            0.46673357486724854
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    [{'test_loss': 0.5618957877159119, 'test_r2': 0.46673357486724854}]

```python
ordered_cols = []
for lag in range (5,0,-1):
    for col in cols:
        ordered_cols.append('{0}_{1}'.format(col , lag))

X = X.reindex(columns=ordered_cols)
X.columns
```

    Index(['DJ_return_5', 'log_volume_5', 'log_volatility_5', 'DJ_return_4',
           'log_volume_4', 'log_volatility_4', 'DJ_return_3', 'log_volume_3',
           'log_volatility_3', 'DJ_return_2', 'log_volume_2', 'log_volatility_2',
           'DJ_return_1', 'log_volume_1', 'log_volatility_1'],
          dtype='object')

```python
X_rnn = X.to_numpy().reshape((-1,5,3))
X_rnn.shape
```

    (6046, 5, 3)

```python
datasets = []
for mask in [train, ~train]:
    X_rnn_t = torch.tensor(
        np.asarray(X_rnn[mask]).astype(np.float32)
    )
    Y_t = torch.tensor(
        np.asarray(Y[mask]).astype(np.float32)
    )
    datasets.append(TensorDataset(X_rnn_t , Y_t))
nyse_train, nyse_test = datasets
```

## RNN
10. In Section 10.9.6, we showed how to fit a linear AR model to the
NYSE data using the LinearRegression() function. However, we also
mentioned that we can “flatten” the short sequences produced for
the RNN model in order to fit a linear AR model. Use this latter
approach to fit a linear AR model to the NYSE data.

```python
class NYSEModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = 0.001
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.train_r2 = R2Score()

        self.rnn = nn.RNN(3, 12, batch_first=True)
        self.dense = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        val, h_n = self.rnn(x)
        val = self.dense(val[:,-1])
        return torch.flatten(val)
        
    def _shared_step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        self.train_r2.update(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        self.val_r2.update(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_r2.update(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return RMSprop(self.parameters(), lr=self.learning_rate)
```

```python
nyse_model = NYSEModel()
nyse_dm = AutoRegressionDataModule(nyse_train, nyse_test, batch_size=64)
nyse_logger = TensorBoardLogger('logs', name='NYSE_RNN')
nyse_trainer = Trainer(
    deterministic=True, 
    max_epochs=200, 
    logger=nyse_logger,
    callbacks=EarlyStopping(monitor='val_loss', patience=5, mode='min')
)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
nyse_trainer.fit(nyse_model, datamodule=nyse_dm)
```
    
      | Name     | Type       | Params | Mode 
    ------------------------------------------------
    0 | loss_fn  | MSELoss    | 0      | train
    1 | val_r2   | R2Score    | 0      | train
    2 | test_r2  | R2Score    | 0      | train
    3 | train_r2 | R2Score    | 0      | train
    4 | rnn      | RNN        | 204    | train
    5 | dense    | Sequential | 13     | train
    ------------------------------------------------
    217       Trainable params
    0         Non-trainable params
    217       Total params
    0.001     Total estimated model params size (MB)
    8         Modules in train mode
    0         Modules in eval mode

#### Note 
This one underperform (only 0.4 R2 score) because we dont fit the dayofweek and month in this model. The conclusion is even the more complex model cant outperform without more context/quality on data.

```python
nyse_trainer.test(nyse_model, datamodule=nyse_dm)
```

    Testing: |                                                | 0/? [00:00<?, ?it/s]

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_loss           0.6231500506401062
             test_r2            0.4086001515388489
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    [{'test_loss': 0.6231500506401062, 'test_r2': 0.4086001515388489}]

## Fusion RNN + Dense
This model inspired from CNN that has its own conv layer and dense layer. The input in dense layer will be the output of LSTM, dayofweek, and month

```python
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, X_seq, X_tab, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)        # shape: (N, T, F)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)        # shape: (N, D)
        self.y = torch.tensor(y, dtype=torch.float32)                # shape: (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_tab[idx], self.y[idx]

class FusionModel(pl.LightningModule):
    def __init__(self, tabular_dim):  # jumlah fitur tabular (D)
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = 0.001
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        self.lstm = nn.LSTM(3, 12, batch_first=True)

        self.dense = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(12 + tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_tabular):
        val, _ = self.lstm(x_seq)              # val: (batch, seq_len, 12)
        val_last = val[:, -1, :]              # ambil timestep terakhir
        combined = torch.cat([val_last, x_tabular], dim=1)  # (batch, 12 + D)
        out = self.dense(combined)
        return torch.flatten(out)

    def _shared_step(self, batch):
        x_seq, x_tabular, y = batch
        preds = self(x_seq, x_tabular)
        loss = self.loss_fn(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        self.train_r2.update(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_r2.update(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        self.test_r2.update(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return RMSprop(self.parameters(), lr=self.learning_rate)

class FusionDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset=None,
                 batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset or val_dataset  # fallback
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

```

```python
X_tab = X.join(NYSE["day_of_week"])
X_tab['month'] = X_tmp.index.month
X_tab = X_tab[['day_of_week', 'month']]
X_tab = pd.get_dummies(X_tab, columns=["day_of_week", "month"])
X_tab = X_tab.to_numpy()
X_tab.shape, X_rnn.shape
```

    ((6046, 17), (6046, 5, 3))

```python
X_rnn[train].shape, X_tab[train].shape
```

    ((4276, 5, 3), (4276, 17))

```python
train_ds = FusionDataset(X_rnn[train], X_tab[train], Y[train])
val_ds = FusionDataset(X_rnn[~train], X_tab[~train], Y[~train])

dm = FusionDataModule(train_ds, val_ds, batch_size=64)

model = FusionModel(tabular_dim=X_tab.shape[1])

trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, dm)
```

    /tmp/ipykernel_571680/3164401403.py:7: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      self.y = torch.tensor(y, dtype=torch.float32)                # shape: (N,)
    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    
      | Name     | Type       | Params | Mode 
    ------------------------------------------------
    0 | loss_fn  | MSELoss    | 0      | train
    1 | train_r2 | R2Score    | 0      | train
    2 | val_r2   | R2Score    | 0      | train
    3 | test_r2  | R2Score    | 0      | train
    4 | lstm     | LSTM       | 816    | train
    5 | dense    | Sequential | 993    | train
    ------------------------------------------------
    1.8 K     Trainable params
    0         Non-trainable params
    1.8 K     Total params
    0.007     Total estimated model params size (MB)
    11        Modules in train mode
    0         Modules in eval mode

    `Trainer.fit` stopped: `max_epochs=50` reached.

```python
trainer.test(model, dm)
```

    Testing: |                                                | 0/? [00:00<?, ?it/s]

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_loss           0.5550546646118164
             test_r2            0.47322607040405273
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    [{'test_loss': 0.5550546646118164, 'test_r2': 0.47322607040405273}]

```python

```

```python

```

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
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
```

```python
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
```

```python
d = load_data('Default')
d.head(), d.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype   
    ---  ------   --------------  -----   
     0   default  10000 non-null  category
     1   student  10000 non-null  category
     2   balance  10000 non-null  float64 
     3   income   10000 non-null  float64 
    dtypes: category(2), float64(2)
    memory usage: 176.2 KB

    (  default student      balance        income
     0      No      No   729.526495  44361.625074
     1      No     Yes   817.180407  12106.134700
     2      No      No  1073.549164  31767.138947
     3      No      No   529.250605  35704.493935
     4      No      No   785.655883  38463.495879,
     None)

```python
default = pd.get_dummies(d, columns=['default', 'student'], drop_first=True)
X = default.drop(columns=['default_Yes'])
y = default['default_Yes']
```

```python
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, shuffle=True, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
```

```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
(lr_pred == y_test).mean()
```

    0.9708

```python
X_train_t = torch.tensor(X_train.astype(np.float32))
X_test_t = torch.tensor(X_test.astype(np.float32))
y_train_t = torch.tensor(y_train.to_numpy().astype(np.float32))
y_test_t = torch.tensor(y_test.to_numpy().astype(np.float32))

default_train = TensorDataset(X_train_t, y_train_t)
default_test = TensorDataset(X_test_t, y_test_t)
```

```python
class DefaultModel(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()
        self.learning_rate = 0.001
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def step(self, batch, batch_idx, mode='train'):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f'{mode}_acc', self.accuracy(preds, y), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 class DefaultModel(pl.LightningModule):
          2     def __init__(self, input_size):
          3         super().__init__()

    NameError: name 'pl' is not defined

```python
class DefaultDataModule(pl.LightningDataModule):
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
default_model = DefaultModel(X_train_t.shape[1])
default_dm = DefaultDataModule(default_train, default_test)
default_logger = TensorBoardLogger('logs', name='default')
default_trainer = Trainer(
    deterministic=True, 
    max_epochs=30, 
    logger=default_logger,
    callbacks=EarlyStopping(monitor='val_acc', patience=5, mode='min'))
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
default_trainer.fit(default_model, datamodule=default_dm) 
```
    
      | Name     | Type              | Params | Mode 
    -------------------------------------------------------
    0 | loss_fn  | BCEWithLogitsLoss | 0      | train
    1 | accuracy | BinaryAccuracy    | 0      | train
    2 | model    | Sequential        | 51     | train
    -------------------------------------------------------
    51        Trainable params
    0         Non-trainable params
    51        Total params
    0.000     Total estimated model params size (MB)
    7         Modules in train mode
    0         Modules in eval mode

```python
default_trainer.test(default_model, datamodule=default_dm)
```

    Testing: |                                                | 0/? [00:00<?, ?it/s]

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_acc            0.9616000056266785
            test_loss           0.10796656459569931
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    [{'test_loss': 0.10796656459569931, 'test_acc': 0.9616000056266785}]

```python

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
      <th>balance</th>
      <th>income</th>
      <th>student_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>729.526495</td>
      <td>44361.625074</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>817.180407</td>
      <td>12106.134700</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1073.549164</td>
      <td>31767.138947</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>529.250605</td>
      <td>35704.493935</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>785.655883</td>
      <td>38463.495879</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>711.555020</td>
      <td>52992.378914</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>757.962918</td>
      <td>19660.721768</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>845.411989</td>
      <td>58636.156984</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>1569.009053</td>
      <td>36669.112365</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>200.922183</td>
      <td>16862.952321</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 3 columns</p>
</div>

```python

```

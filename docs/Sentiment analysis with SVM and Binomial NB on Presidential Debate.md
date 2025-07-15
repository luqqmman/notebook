# Sentiment analysis with Logistic Regression, SVM, Binomial Naive Bayes, and Deep Learning on Presidential Debate
This exercise derived from Data Mining project in my college but i remaster it with SVM and Deep Learning. My team scraped and manually label the data before, so we just need to do a little preprocess here.

```python
import pandas as pd
```

```python
pd.set_option('display.max_colwidth', None)
df = pd.read_csv("dataset/president_debate.csv")
df = df.dropna()
df.head()
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
      <th>label</th>
      <th>full_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>Ganjar menyebut pernyataan itu disampaikan Jokowi saat debat Capres di tahun 2019 silam. Saat itu Ganjar merupakan salah satu tim kampanye Jokowi. TAG: Jokowi | Ganjar | anies final stage | prabowo | all in 02 | Ketua KPU | Australia | agak laen | pemilu 2024 | pemilu 2019 https://t.co/QuEav9i1Bw</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>@pikiranlugu @99propaganda @bengkeldodo @Ndons_Back @BANGSAygSUJUD @_NusantaraLand_ @florieliciouss @are_inismyname @Reskiichsan8 @P4P4B0W0_2024 @AditBandit234 @kurawa Sadar lah kamu Erik jangan nyebar in isu murahan seperti itu..saya juga nonton debat terahir V dan Pak ANIS CAPRES 01 tidak punya Niat ataupun bicara akan meruba BUMN menjadi KOPERASI..VISI MISI pak ANIS RB Jelas..</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>@WagimanDeep212_ Dri debat terakhir kemarin nih. Makin mantep dan yakin pilih ganjar mahfud. Gaspolll ykin m3nang</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>@Dy_NPR Gak perlu kami prihatin dg modelan begini. Sdh sepuh kesehatan entah. Untungnya debat terakhir P Anies berwelas asih dg tidak membuat beliau berkaca2 lagi. Kpn nabgis massal? Bnyk yg nungguin nih https://t.co/e5A9ihPqLP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>@tempodotco Selalu suka sama pembawaannya pak Anies yang adem apalagi pas debat semalem beliau keliatan tenang dan sudah mempersiapkan diri banget yuk bisa AMIN 1 putaran aja https://t.co/5RtE6Zop33</td>
    </tr>
  </tbody>
</table>
</div>

```python
df['label'].value_counts()
```

    label
    1.0    681
    0.0    337
    Name: count, dtype: int64

## First we downsample the data to stratify the class

```python
positive = df[df['label'] == 1].sample(337, random_state=0)
negative = df[df['label'] == 0]
df = pd.concat([positive, negative])
df.label.value_counts()
```

    label
    1.0    337
    0.0    337
    Name: count, dtype: int64

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
```

    [nltk_data] Downloading package punkt_tab to /home/luqman/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!
    [nltk_data] Downloading package punkt to /home/luqman/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /home/luqman/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!

    True

## Next we clean the data
Tweet has username, tag, and links, we will get rid of that. We also need to remove stopword because it not helping us to determine the sentiment so basically reduce the dimension of the model.

```python
def preprocess(text):
    text = text.lower()
    ## remove url
    text = re.sub(r'http\S+|www.\S+', '', text)
    ## remove username dan hastag
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    ## remove non ascii
    text = text.encode('ascii', 'ignore').decode('utf-8')
    ## remove number and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    ## remove stop word
    tokens = word_tokenize(text)
    st = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in st]
    return ' '.join(tokens)

df['full_text'] = df['full_text'].apply(preprocess)
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'], stratify=df['label'], shuffle=True, random_state=0)
```

## Next we do vectorization with tfidf
It similar to bag of word but instead of count we do tf*idf which term frequency(simplest: count of that word in that document) * inverse document frequency(how much document have that word accross all documents). When the word is rare the weight is higher, tfidf suit our need to measure different word on how important that word is, so we can decide positive or negative sentiment better. From the shape of our training data, we have 2302 word and 

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
X_train.shape
```

    (505, 2302)

```python
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

## Comparing model
we will compare the first 3 models then compare them with deep learning after quick evaluation
 - linear SVM (draw decision line counting only closer/support data)
 - naive bayes (use bayes theorem with assumption all word independent and based on bernoulli/binomial distribution)
 - Logistic regression (modelling logit of sentiment with maximum likelihood estimation)
 - Deep Learning (neural network with hidden layer to extract feature and do logistic regression on the extracted feature) 

```python
linsvm = SVC(kernel='linear')
linsvm.fit(X_train, y_train)
pred_linsvm = linsvm.predict(X_test)
print(classification_report(y_test, pred_linsvm), accuracy_score(y_test, pred_linsvm))
```

                  precision    recall  f1-score   support
    
             0.0       0.85      0.88      0.87        85
             1.0       0.88      0.85      0.86        84
    
        accuracy                           0.86       169
       macro avg       0.86      0.86      0.86       169
    weighted avg       0.86      0.86      0.86       169
     0.863905325443787

```python
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
pred_bnb = bnb.predict(X_test)
print(classification_report(y_test, pred_bnb), accuracy_score(y_test, pred_bnb))
```

                  precision    recall  f1-score   support
    
             0.0       0.95      0.66      0.78        85
             1.0       0.74      0.96      0.84        84
    
        accuracy                           0.81       169
       macro avg       0.84      0.81      0.81       169
    weighted avg       0.84      0.81      0.81       169
     0.8106508875739645

```python
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred_mnb = mnb.predict(X_test)
print(classification_report(y_test, pred_mnb), accuracy_score(y_test, pred_mnb))
```

                  precision    recall  f1-score   support
    
             0.0       0.88      0.84      0.86        85
             1.0       0.84      0.88      0.86        84
    
        accuracy                           0.86       169
       macro avg       0.86      0.86      0.86       169
    weighted avg       0.86      0.86      0.86       169
     0.8579881656804734

```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print(classification_report(y_test, pred_lr), accuracy_score(y_test, pred_lr))
```

                  precision    recall  f1-score   support
    
             0.0       0.85      0.86      0.85        85
             1.0       0.86      0.85      0.85        84
    
        accuracy                           0.85       169
       macro avg       0.85      0.85      0.85       169
    weighted avg       0.85      0.85      0.85       169
     0.8520710059171598

## What we got, before continuing to deep learning
- From SVM results, it looks like the sentiment is linearly separable so maybe complex model like deep learning won't necessary
- Logistic regression offer balance between accuracy and interpretation, we can select most positive/negative weight to see which word is mostly determine positve and negative sentiment.

#### Most "sentiment" word according to Logistic Regression weight

```python
feature_names = tf.get_feature_names_out()
coefficients = lr.coef_[0] 

# Top word for positif and negative sentiment
top_pos = sorted(zip(coefficients, feature_names), reverse=True)[:10]
top_neg = sorted(zip(coefficients, feature_names))[:10]

print("Top positive words:")
for coef, word in top_pos:
    print(f"{word}: {coef:.4f}")

print("\nTop negative words:")
for coef, word in top_neg:
    print(f"{word}: {coef:.4f}")
```

    Top positive words:
    ganjar: 2.2100
    ganjarmahfud: 1.9752
    pranowo: 1.6685
    mahfud: 1.3405
    malam: 1.1633
    terbaik: 1.0706
    banget: 1.0337
    indonesia: 0.9939
    keren: 0.9842
    rakyat: 0.9240
    
    Top negative words:
    bansos: -1.7470
    yg: -1.2866
    ga: -1.2157
    jokowi: -1.0811
    prabowo: -0.9891
    covid: -0.9811
    aja: -0.9744
    bermasalah: -0.9656
    dikorupsi: -0.9656
    kaesang: -0.9656

## Deep learning
We will try deep learning using 1 hidden layer with 64 hidden unit, effectively try to reduce the dimension before doing logistic regression in the output layer. We will use the test set as validation.

```python
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
from ISLP import load_data
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
class SentimentModel(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.learning_rate = 0.001
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def _shared_step(self, batch, stage):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        
        if stage == 'train':
            metric = self.train_accuracy
        elif stage == 'val':
            metric = self.val_accuracy
        else:
            metric = self.test_accuracy
        metric(preds, y.float())
        self.log(f'{stage}_loss', loss, on_epoch=True, on_step=False)
        self.log(f'{stage}_acc', metric, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, 'test')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

class SentimentDataModule(pl.LightningDataModule):
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
X_train_t = torch.tensor(X_train.toarray().astype(np.float32))
y_train_t = torch.tensor(y_train.to_numpy().astype(np.float32))
X_test_t = torch.tensor(X_test.toarray().astype(np.float32))
y_test_t = torch.tensor(y_test.to_numpy().astype(np.float32))

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)
```

```python
sentiment_model = SentimentModel(X_train_t.shape[1])
sentiment_dm = SentimentDataModule(train_ds, test_ds)
sentiment_logger = CSVLogger('logs', name='sentiment')
sentiment_trainer = Trainer(
    deterministic=True, 
    max_epochs=30, 
    logger=sentiment_logger,
    callbacks=EarlyStopping(monitor='val_acc', patience=5, mode='min')
)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs

```python
sentiment_trainer.fit(sentiment_model, sentiment_dm)
```
    
      | Name           | Type              | Params | Mode 
    -------------------------------------------------------------
    0 | loss_fn        | BCEWithLogitsLoss | 0      | train
    1 | train_accuracy | BinaryAccuracy    | 0      | train
    2 | val_accuracy   | BinaryAccuracy    | 0      | train
    3 | test_accuracy  | BinaryAccuracy    | 0      | train
    4 | model          | Sequential        | 147 K  | train
    -------------------------------------------------------------
    147 K     Trainable params
    0         Non-trainable params
    147 K     Total params
    0.590     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode

    /home/luqman/Lab/islp/venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (16) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.

```python
sentiment_trainer.test(sentiment_model, sentiment_dm)
```

    Testing: |                                                                                                    …

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_acc            0.8461538553237915
            test_loss           0.5141122341156006
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    [{'test_loss': 0.5141122341156006, 'test_acc': 0.8461538553237915}]

```python
sentiment_results = pd.read_csv(sentiment_logger.experiment.metrics_file_path)

fig, ax = subplots()
sentiment_results.dropna(subset='train_acc').plot('epoch', 'train_acc', ax=ax)
sentiment_results.dropna(subset='val_acc').plot('epoch', 'val_acc', ax=ax)
best_val = sentiment_results['val_acc'].max()
ax.axhline(best_val, linestyle='--')
best_val
```

    0.8698225021362305
    
![png](Sentiment%20analysis%20with%20SVM%20and%20Binomial%20NB%20on%20Presidential%20Debate_files/Sentiment%20analysis%20with%20SVM%20and%20Binomial%20NB%20on%20Presidential%20Debate_28_1.png)

## Conclusion
- Deep learning outperform all model with with 87% accuracy at it's peak
- It hard to interpret deep learning so a simpler model like Logistic Regression should be enough for this task with 86% accuracy
- I personally prefer simpler model for this task especially SVM for more accuracy and Logistic regression for interpretability. We have'nt tune the hyperparameter yet and the simpler model might be better after tuning because it easier to train.

```python

```

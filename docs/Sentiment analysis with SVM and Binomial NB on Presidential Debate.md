# Sentiment analysis with SVM and Binomial NB on Presidential Debate
This exercise derived from Data Mining project in my college but i improvise with SVM. My team scraped and manually label the data before, so we just need to do some preprocess.


```python
import pandas as pd
```


```python
pd.set_option('display.max_colwidth', None)
df = pd.read_csv("dataset/president_debate.csv")
df = df.dropna()
df
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>1.0</td>
      <td>Tetapi itikad kita baik saya kira 3 paslon semuanya ingin yang terbaik untuk rakyat Indonesia kata Prabowo dalam pernyataan penutupnya di sesi debat kelima Pilpres 2024 yang digelar di Jakarta Convention Center (JCC) Jakarta Minggu (4/2).</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>1.0</td>
      <td>@kompascom @aniesbaswedan Pak Anies berhasil bikin debat semalam jadi punya nuansa berbeda keren parah!</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>1.0</td>
      <td>@kompascom @aniesbaswedan Debat semalam semakin berkesan dengan pesona dan kata-kata positif dari Pak Anies!</td>
    </tr>
    <tr>
      <th>1067</th>
      <td>1.0</td>
      <td>@IDNTimes kalo melihat dari debat capres terakhir kemarin aku bakalan mantepin hatiku buat pilih Ganjar-Mahfud</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>1.0</td>
      <td>@raffimulyaa Nah kalo menurut orang2 debat kemarin gak asik krn ga panas. Menurut gue pak Anies pake strategi yang bagus buat debat terakhir ini. Ada remnya dan mengedepankan apa yang bakal dilakuin sebagai calon presiden kedepan. Masalahnya kl ngegoreng kyk sblmnya dah kelar kewarasan kita</td>
    </tr>
  </tbody>
</table>
<p>1018 rows × 2 columns</p>
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
    [nltk_data]   Unzipping tokenizers/punkt_tab.zip.
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
    ## hapus url
    text = re.sub(r'http\S+|www.\S+', '', text)
    ## hapus username dan hastag
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    ## hapus non ascii
    text = text.encode('ascii', 'ignore').decode('utf-8')
    ## hapus angka dan tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    ## hapus stop word
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
It similar to bag of word but instead of count we do tf*idf which term frequency(simplest: count of that word in that document) * inverse document frequency(how much document have that word accross all documents). When the word is rare the weight is higher, tfidf suit our need to measure different word on how important that word is, so we can decide positive or negative sentiment better.  


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
X_train.shape
```




    (763, 4023)




```python
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

## Comparing model
we compare linear SVM (draw decision line counting only closer/support data), naive bayes (use bayes theorem with assumption all word independent and based on binomial/bernoulli distribution), dan Logistic regression (similar to bernoulli NB but instead using bayes theorem we use gradient descent to model the mean)
- It looks like the sentiment is linearly separable and linear svm outperform for this random seed data
- BernoulliNB offer more precision with recall trade off
- Logistic regression easiest to interpret, we can select most positive/negative weight to see which word is mostly determine positve and negative sentiment.

### based on logistic regression

Top positive words:
- ganjar: 1.9254
- mahfud: 1.9136
- mohmahfudmd: 1.4525
- aniesbaswedan: 1.325

Top negative words:
- prabowo: -2.0228
- bansos: -1.9204
- apa: -1.4161
- jokowi: -1.1784


```python
linsvm = SVC(kernel='linear')
linsvm.fit(X_train, y_train)
pred_linsvm = linsvm.predict(X_test)
print(classification_report(y_test, pred_linsvm), accuracy_score(y_test, pred_linsvm))
```

                  precision    recall  f1-score   support
    
             0.0       0.86      0.88      0.87        85
             1.0       0.88      0.86      0.87        84
    
        accuracy                           0.87       169
       macro avg       0.87      0.87      0.87       169
    weighted avg       0.87      0.87      0.87       169
     0.8698224852071006



```python
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
pred_bnb = bnb.predict(X_test)
print(classification_report(y_test, pred_bnb), accuracy_score(y_test, pred_bnb))
```

                  precision    recall  f1-score   support
    
             0.0       0.91      0.68      0.78        85
             1.0       0.74      0.93      0.83        84
    
        accuracy                           0.80       169
       macro avg       0.82      0.81      0.80       169
    weighted avg       0.83      0.80      0.80       169
     0.8047337278106509



```python
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred_mnb = mnb.predict(X_test)
print(classification_report(y_test, pred_mnb), accuracy_score(y_test, pred_mnb))
```

                  precision    recall  f1-score   support
    
             0.0       0.85      0.85      0.85        85
             1.0       0.85      0.85      0.85        84
    
        accuracy                           0.85       169
       macro avg       0.85      0.85      0.85       169
    weighted avg       0.85      0.85      0.85       169
     0.8461538461538461



```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print(classification_report(y_test, pred_lr), accuracy_score(y_test, pred_lr))
```

                  precision    recall  f1-score   support
    
             0.0       0.83      0.65      0.73        84
             1.0       0.85      0.94      0.89       171
    
        accuracy                           0.84       255
       macro avg       0.84      0.80      0.81       255
    weighted avg       0.84      0.84      0.84       255
     0.8431372549019608


## Most "sentiment" word


```python
feature_names = tf.get_feature_names_out()
coefficients = lr.coef_[0]  # hanya 1 baris untuk binary classification

# Top kata positif dan negatif
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
    ganjar: 1.9254
    mahfud: 1.9136
    mohmahfudmd: 1.4525
    aniesbaswedan: 1.3253
    co: 1.3187
    pak: 1.3028
    https: 1.3028
    ganjarmahfud2024: 1.2753
    ganjarpranowo: 1.2635
    yang: 1.2433
    
    Top negative words:
    prabowo: -2.0228
    bansos: -1.9204
    apa: -1.4161
    jokowi: -1.1784
    aja: -1.1065
    ga: -1.0893
    covid: -1.0787
    19: -1.0573
    bermasalah: -1.0573
    dikorupsi: -1.0573



```python

```

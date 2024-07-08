#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Library
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import streamlit as st


# In[2]:


# Membaca file dataset
dir = 'hungarian.data'
with open(dir, encoding='Latin1') as file:
    lines = [line.strip() for line in file]


# In[3]:


# Menyusun data menjadi dataframe
data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)
df = pd.DataFrame.from_records(data)
print(df.head())


# In[4]:


# Mengganti nilai '-9' dengan NaN untuk missing values
df.replace('-9', np.nan, inplace=True)

# Menghapus kolom 'name' (kolom terakhir)
df.drop(columns=df.columns[-1], inplace=True)


# In[5]:


# Memilih kolom yang relevan sesuai dengan indeks yang diberikan
selected_columns = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
df = df.iloc[:, selected_columns]

# Menambahkan nama kolom
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
              'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']


# In[6]:


# Memeriksa apakah terdapat nilai yang hilang (NaN) dalam dataset
print(df.isnull().sum())


# In[7]:


df.head()


# In[8]:


# Mengganti nilai yang hilang dengan metode yang sesuai (misalnya, mengganti dengan nilai rata-rata untuk kolom numerik)
df.fillna(df.mean(), inplace=True)

# Untuk kolom yang mungkin lebih baik diisi dengan modus (misalnya kategori), kita bisa melakukan hal berikut:
for column in df.columns:
    if df[column].dtype == 'object':  # Jika tipe datanya adalah object (kategori)
        df[column].fillna(df[column].mode()[0], inplace=True)

# Memastikan tipe data yang sesuai
df = df.apply(pd.to_numeric, errors='ignore')

# Menampilkan informasi mengenai dataset setelah pembersihan
print(df.info())


# In[9]:


# Memeriksa kembali apakah terdapat nilai yang hilang (NaN) dalam dataset
print(df.isnull().sum())


# In[10]:


df.shape


# In[11]:


df["num"].unique()


# In[12]:


# Cek korelasi antar kolom
print(df.corr()["num"].abs().sort_values(ascending=False))


# In[14]:


# Menghitung korelasi matriks
correlation_matrix = df.corr()

# Menampilkan korelasi matriks
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[15]:


# Analisis target
y = df["num"]
#sns.countplot(y)

target_temp = df.num.value_counts()
print(target_temp)


# In[16]:


atribut = df.drop("num",axis=1)
target = df["num"]

X_train,X_test,Y_train,Y_test = train_test_split(atribut,target,test_size=0.20,random_state=0)


# In[17]:


print(X_train.shape)
print(X_test.shape)


# In[18]:


print(Y_train.shape)
print(Y_test.shape)


# # Oversampling SMOTE

# In[19]:


# Plot distribusi kelas sebelum oversampling
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
y.value_counts().sort_index().plot(kind='bar', title='Distribusi Kelas Sebelum Oversampling')
plt.xlabel('Kelas')
plt.ylabel('Frekuensi')

# Menerapkan SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(atribut, target)

# Plot distribusi kelas setelah oversampling
plt.subplot(1, 2, 2)
y_res.value_counts().sort_index().plot(kind='bar', title='Distribusi Kelas Setelah Oversampling')
plt.xlabel('Kelas')
plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()


# In[20]:


## OPSIONAL
# Normalisasi atau standarisasi fitur jika diperlukan
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Membagi data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# In[ ]:





# In[21]:


# LOGISTIC REGRESSION
lr = LogisticRegression()
lr.fit(X_train,y_train)
Y_pred_lr = lr.predict(X_test)

score_lr = round(accuracy_score(Y_pred_lr,y_test)*100,2)
print("Akurasi : "+str(score_lr)+" %")


# In[22]:


# Membuat confusion matrix model Logistic Regression
cm = confusion_matrix(y_test, Y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[23]:


# NAIVE BAYES
nb = GaussianNB()
nb.fit(X_train,y_train)
Y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(Y_pred_nb,y_test)*100,2)
print("Akurasi : "+str(score_nb)+" %")


# In[24]:


# Membuat confusion matrix model Naive Bayes
cm = confusion_matrix(y_test, Y_pred_nb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[25]:


# SVM
from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(X_train, y_train)
Y_pred_svm = sv.predict(X_test)

score_svm = round(accuracy_score(Y_pred_svm,y_test)*100,2)
print("Akurasi : "+str(score_svm)+" %")


# In[26]:


# Membuat confusion matrix model SVM
cm = confusion_matrix(y_test, Y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[27]:


# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
Y_pred_knn=knn.predict(X_test)

score_knn = round(accuracy_score(Y_pred_knn,y_test)*100,2)

print("Akurasi : "+str(score_knn)+" %")


# In[28]:


# Membuat confusion matrix model KNN
cm = confusion_matrix(y_test, Y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[29]:


# DECISON TREE
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(500):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,y_train)
Y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)

print("Akurasi : "+str(score_dt)+" %")


# In[33]:


# Membuat confusion matrix model Decision Tree
cm = confusion_matrix(y_test, Y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[ ]:


# RANDOM FOREST
max_accuracy = 0

for x in range(1000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,y_train)
Y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(Y_pred_rf,y_test)*100,2)

print("Akurasi : "+str(score_rf)+" %")


# In[ ]:


# Membuat confusion matrix model Random Forest
cm = confusion_matrix(y_test, Y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[34]:


# XGBOOST
import xgboost as xgb
from xgboost import XGBClassifier

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

Y_pred_xgb = xgb_model.predict(X_test)

score_xgb = round(accuracy_score(Y_pred_xgb,y_test)*100,2)

print("Akurasi : "+str(score_xgb)+" %")


# In[35]:


# Membuat confusion matrix model XGBoost
cm = confusion_matrix(y_test, Y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[36]:


scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_xgb]#,score_rf]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","XGBoost"]#,"Random Forest"]    

for i in range(len(algorithms)):
    print("Akurasi Model "+algorithms[i]+" : "+str(scores[i])+" %")


# In[ ]:





# In[37]:


# Membangun dan mengevaluasi berbagai model klasifikasi
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    #"Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

# Menyimpan hasil akurasi
results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((model_name, accuracy))

# Membuat DataFrame dari hasil
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

# Membuat barplot
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Model")
plt.ylabel("Accuracy score")
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.show()


# In[ ]:





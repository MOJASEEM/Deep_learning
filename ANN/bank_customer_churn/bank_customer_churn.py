import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
df = pd.read_csv("D:/Data Science/customer_churn_prediction/bank_customer_churn/Churn_Modelling_bank.csv")
print(df.sample(5))
print(df.dtypes)
print(df['Exited'].value_counts())
exited_no = df[df.Exited==0].CreditScore
exited_yes = df[df.Exited==1].CreditScore
plt.xlabel("Credit Score")
plt.ylabel("Number Of Customers")
plt.title("Bank Customer Churn Prediction Visualization")
plt.hist([exited_yes, exited_no], rwidth=0.95, color=['green','red'],label=['Exited=Yes','Exited=No'])
plt.legend()
plt.show()
def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}')
print_unique_col_values(df)
df1 = df.drop(['RowNumber','CustomerId','Surname'],axis='columns')
print(df1.dtypes)
df1 = pd.get_dummies(data=df1,columns=['Geography','Gender'],drop_first=True)
print(df1.dtypes)
scaler = MinMaxScaler()
numerical_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])
print(df1.sample(5))
X = df1.drop('Exited',axis='columns')
y = df1.Exited
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
print(X_train.shape)    
print(X_test.shape)
model = keras.Sequential([
    keras.layers.Dense(20,input_shape=(X_train.shape[1],),activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)
model.evaluate(X_test,y_test)
y_p = model.predict(X_test)
y_p = y_p.flatten()
y_pred = []
for element in y_p:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(classification_report(y_test,y_pred))
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


import pandas as pd
import numpy as nm

data = pd.read_csv('bill_authentication.csv')
print(data)

x=data.iloc[:,0:4].values
print(x)
y=data.iloc[:,4].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
Classifier= RandomForestClassifier(n_estimators= 10, random_state=0)
Classifier.fit(x_train,y_train)

y_pred = Classifier.predict(x_test)

# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_test,y_pred)
# print(cm)

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))
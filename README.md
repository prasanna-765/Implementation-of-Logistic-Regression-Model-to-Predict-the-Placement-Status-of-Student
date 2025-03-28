# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
    

## Program:
## Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
### Developed by: Prasanna A
### RegisterNumber:  23005675
```
import pandas as pd

data=pd.read_csv(r"Placement_Data.csv")

data.head()

data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis=1)

data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])

data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"])

data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"])

data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])

data1["status"]=le.fit_transform(data1["status"])

data1

x=data1.iloc[:,:-1]

x

y=data1["status"]

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver="liblinear")

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

accuracy

from sklearn.metrics import confusion_matrix


confusion=confusion_matrix(y_test,y_pred)

confusion

from sklearn.metrics import classification_report

classification_report1=classification_report(y_test,y_pred)

print(classification_report1)
 ```
## Output:
![Screenshot 2024-11-22 090225](https://github.com/user-attachments/assets/26aa1f1c-a3a0-4151-b9f5-959911fac3d9)

![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot 2024-11-22 0904341](https://github.com/user-attachments/assets/aa35ba0c-f7f6-4713-8d4c-1dfc57314dff)
![Screenshot 2024-11-22 090434](https://github.com/user-attachments/assets/5e3df1b8-e6a5-48ca-b3c8-42cbe601b4d5)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vedagiri Indu Sree
RegisterNumber:  212223230236
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/user-attachments/assets/b3d0b8f2-73a4-4716-841a-29e05b3b92b0)

![image](https://github.com/user-attachments/assets/587fd560-42ee-4940-abeb-8220822aee04)

![image](https://github.com/user-attachments/assets/e465206a-0e48-48bd-9f02-8dfcbdf12717)

![image](https://github.com/user-attachments/assets/451eed50-2906-49d6-a8df-22e594bb4c60)

![image](https://github.com/user-attachments/assets/b0f9d6b0-aafc-4ffc-b503-657ee5aaa372)

![image](https://github.com/user-attachments/assets/818c9cb1-c089-462b-9021-b8bc27a3b25b)

![image](https://github.com/user-attachments/assets/257af569-982a-42d2-896f-17d2de7e33d2)

![image](https://github.com/user-attachments/assets/cf5fd279-f982-40cd-aad0-217f67c28b53)

![image](https://github.com/user-attachments/assets/94fe16e3-c72a-4d40-a909-af97b6188f4c)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

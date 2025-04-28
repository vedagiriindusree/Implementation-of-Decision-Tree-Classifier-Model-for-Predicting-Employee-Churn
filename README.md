# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not. 

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
print("Name:Vedagiri Indu sree")
print("Reg No:212223230236")
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

![image](https://github.com/user-attachments/assets/02f7721b-be89-4e85-af6a-c70f9b7d3d34)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

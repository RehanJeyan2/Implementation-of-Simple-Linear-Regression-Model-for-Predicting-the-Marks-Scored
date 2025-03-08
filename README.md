# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rehan Jeyan
RegisterNumber:  212223040167
*/
```

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```
<br>
<br>
<br>
<br>
<br>
<br>


## Output:

### Head Values
![image](https://github.com/user-attachments/assets/e112cae5-dd33-4bdf-8f6f-7411bddda954)


### Tail Values
![image](https://github.com/user-attachments/assets/816c2518-a901-4f34-aa01-905f7ab20c1e)




### Compare Dataset
![image](https://github.com/user-attachments/assets/22bfd98d-4acc-478f-a059-5838d6552b3d)



### Predication values of X and Y
![image](https://github.com/user-attachments/assets/e07328bc-cceb-41d1-b6b2-8146994dade7)


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

### Training set
![image](https://github.com/user-attachments/assets/64d67b9e-e02d-465d-a60a-500a978cb065)



### Testing Set
![image](https://github.com/user-attachments/assets/6610877f-81a6-4ad3-a6f6-6d4cf04de84d)

### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/5a78ef60-cf00-4f1d-8b69-9cf3acef2d66)

<br>
<br>
<br>
<br>
<br>

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

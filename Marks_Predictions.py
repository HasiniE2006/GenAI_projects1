import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data={
    'study_hours':[1,2,3,4,5],
    'Attendance':[60,70,80,90,100],
    'bunk':[3,2,4,5,1],
    'Grade':[75,85,95,100,100]
}
df=pd.DataFrame(data)
x=df[['study_hours','Attendance','bunk']]
y=df['Grade']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
#make predictions
y_pred=model.predict(x_test)
#evaluate the model
accuracy_score=model.score(x_test,y_test)
mae=mean_squared_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Accuracy Score:",accuracy_score)
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)
print("R-squared:",r2)
new_data=pd.DataFrame({'study_hours':[6],'Attendance':[90],'bunk':[2]})
predicted_grade=model.predict(new_data)
print("Predicted Grade:",predicted_grade)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load dataset 
df = pd.read_csv('car data.csv')
df.head()
#drop car_name column
df=df.drop(columns=['Car_Name'])
df.head()
#convert categorical into numerical
df_encoded=pd.get_dummies(df, drop_first=True)
df_encoded.head()
x=df_encoded.drop("Selling_Price",axis=1)
y=df_encoded['Selling_Price']
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#train ml model random forest
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(
    n_estimators=200,
    criterion="squared_error",
    random_state=42
)
model.fit(x_train,y_train)
#evaluate the model
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
y_pred=model.predict(x_test)
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_pred)))
r2_score(y_test,y_pred)
def predict_car_price(year,
                      present_price,
                      kms_driven,
                      fuel_type,
                      owner,
                      company,
                      seller_type,
                      transmission):
  input_data=pd.DataFrame({
      "year":[year],
      "present_price":[present_price],
      "kms_driven":[kms_driven],
      "fuel_type":[fuel_type],
      "owner":[owner],
      "company":[company],
      "seller_type":[seller_type],
      "transmission":[transmission]
  })
  input_data=pd.get_dummies(input_data,drop_first=True)
  input_data=input_data.reindex(columns=x.columns,fill_value=0)
  prediction=model.predict(input_data)
  return prediction
#input 
price=predict_car_price(
    year=2019,
    present_price=7.0,
    kms_driven=100000,
    fuel_type="Petrol",
    owner=1,
    company="Maruti",
    seller_type="Individual",
    transmission="Manual"
)
print("Car Price:",price[0])

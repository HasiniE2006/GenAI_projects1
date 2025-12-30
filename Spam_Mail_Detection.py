import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
data = {
	'message': [
	    'you won lottery',
	    'extra income',
	    'won cash prize',
	    'get rich quick',
	    'limited time offer',
	    'Buy 1 get 1 free',
	    'but Lamborghini with 1 rupee',
	    'you won 1 crore coupon',
	    'Congratulations you won lottery',
	    'get loan of 100 crore today',
	    'dont miss this',
	    'join this zoom meeting at 10am',
	    'reminder',
	    'upgrade your plan',
	    'your OTP is',
	    'upgrade your payment method',
	    'Bank statement',
	    'verify its you',
	    'invoice',
	    'payment conformed'
	],
 'label':[
     'spam','spam','spam','spam','spam','spam','spam','spam','spam','spam','spam','Not spam','Not spam',
     'Not spam','Not spam','Not spam','Not spam','Not spam','Not spam','Not spam'
 ]
}
df=pd.DataFrame(data)
df['label']=df['label'].map({'spam':1,'Not spam':0})
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df['message'])
y=df['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
model=MultinomialNB()
model.fit(x_train,y_train)
new_email=['hello']
new_email_vectorized=vectorizer.transform(new_email)
prediction=model.predict(new_email_vectorized)
if prediction==1:
  print("Spam")
else:
  print("Not spam")

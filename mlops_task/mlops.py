#!/usr/bin/env python
# coding: utf-8

# In[1]:

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template

# In[2]:


data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[3]:


regressor = LinearRegression()
# Train the model using the training sets
regressor.fit(X_train, y_train)


# In[4]:


y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R2 score:', r2)


# In[5]:


# Visualize the training set results
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Visualize the test set results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[6]:





# In[7]:


import joblib

# Save the trained model to a file
joblib.dump(regressor, 'linear_regression_model.pkl')


# In[8]:


from flask import Flask, request, jsonify

app = Flask(__name__)


# In[9]:



# Load the trained model
model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/addData', methods=["POST"])
def add_data():
    experience = request.form['experience']
    print(experience)

    exp= float(experience)
   
    X = np.array([[exp]])
    print(X)
    # Make a prediction using the loaded model
    y_pred = model.predict(X)

    # Return the predicted salary as a JSON response
    return jsonify({'salary': y_pred[0]})



# In[10]:


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request1
    input1 = request.get('fname')
    print(input1)
    data = request.get_json()

    # Convert the input data into a numpy array
    X = np.array([data['experience']])

    # Make a prediction using the loaded model
    y_pred = model.predict(X)

    # Return the predicted salary as a JSON response
    return jsonify({'salary': y_pred[0]})

    # return render_template('index.html')


# In[11]:


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")


# In[12]:




url = 'http://localhost:5000/predict'
data = {'experience': 5}
response = requests.post(url, json=data)
print(response.json())


# In[ ]:





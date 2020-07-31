#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


covid = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/covid_19_world.csv')


# In[3]:


covid


# In[4]:


covid.drop('SNo', axis = 1, inplace = True)


# In[5]:


covid.head()


# In[6]:


covid['ObservationDate'] = pd.to_datetime(covid['ObservationDate'])


# In[7]:


covid.head()


# In[8]:


covid_datewise = covid.groupby(['ObservationDate']).agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'})   #agg = aggregate function and sum is also a function


# In[9]:


covid_datewise


# In[10]:


sns.set_style('darkgrid')
plt.figure(figsize=(20,8))
sns.barplot(x=covid_datewise.index.date, y = covid_datewise['Confirmed'])
plt.title("CoVid-19 Confirmed Cases in World", fontsize=20)
plt.xticks(rotation = 90)
plt.show()


# In[11]:


sns.set_style('darkgrid')
plt.figure(figsize=(20,8))
atv_cases=covid_datewise['Confirmed']-covid_datewise['Recovered']-covid_datewise['Deaths']
sns.barplot(x=covid_datewise.index.date, y = atv_cases, palette='YlOrRd')
plt.title("CoVid-19 Active Cases in World", fontsize=20)
plt.ylabel("Active", fontsize=15)
plt.xticks(rotation = 90)
plt.show()


# ## Forecasting

# ## Linear Regression

# In[12]:


covid_datewise['Days'] = covid_datewise.index - covid_datewise.index[0]


# In[13]:


covid_datewise


# In[14]:


covid_datewise['Days'] = covid_datewise['Days'].dt.days


# In[15]:


covid_datewise.head()


# In[16]:


test_ml = covid_datewise.iloc[int(covid_datewise.shape[0]*0.9):]


# In[17]:


test_ml


# In[18]:


train_ml = covid_datewise.iloc[:int(covid_datewise.shape[0]*0.9)]


# In[19]:


train_ml


# In[20]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression(normalize=True)


# In[21]:


X = np.array(train_ml['Days']).reshape(-1,1)


# In[22]:


X


# In[23]:


Y = np.array(train_ml['Confirmed']).reshape(-1,1)  #to fit our model in a 2D array instead of scalar list


# In[24]:


Y


# In[25]:


#linear regression trained model
lin_reg.fit(X,Y)


# In[26]:


#Prediction
predicted_value = lin_reg.predict(np.array(test_ml['Days']).reshape(-1,1))
predicted_value


# In[27]:


predicted_value_all = lin_reg.predict(np.array(covid_datewise['Days']).reshape(-1,1))


# In[28]:


#Plotting the predicted vs Actual Value
plt.figure(figsize=(15,8))
plt.plot(covid_datewise['Confirmed'], label = 'Confirmed')
plt.plot(covid_datewise.index, predicted_value_all, linestyle = '--', color= 'red', label = 'Predicted Cases')
plt.legend()
plt.show()


# In[29]:


lin_reg.score(np.array(test_ml['Confirmed']).reshape(-1,1), predicted_value) #score is the accuracy rate and it is very bad here


# ## FBProphet - ARIMA(Auto Regressive Integrated Moving Average)

# In[30]:


from fbprophet import Prophet


# In[31]:


prophet_model = Prophet(interval_width=0.95, weekly_seasonality=True, daily_seasonality=True)


# In[32]:


## input samples for fbprophet
prophet_confirmed = pd.DataFrame(zip(list(covid_datewise.index), list(covid_datewise['Confirmed'])), columns=['ds', 'y'])


# In[33]:


prophet_confirmed


# In[35]:


prophet_model.fit(prophet_confirmed)


# In[40]:


forecast_c = prophet_model.make_future_dataframe(periods=18)


# In[41]:


forecast_c


# In[42]:


confirmed_forecast = prophet_model.predict(forecast_c)


# In[43]:


confirmed_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# In[45]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(covid_datewise ["Confirmed"],confirmed_forecast['yhat'].head(covid_datewise.shape[0])))
print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(covid_datewise["Confirmed"],confirmed_forecast['yhat'].head(covid_datewise.shape[0]))))


# In[46]:


print(prophet_model.plot(confirmed_forecast))


# ## LSTM - RNN

# In[49]:


## LSTM - Part RNN with Backpropagation & Previous Memory


# In[51]:


from keras.models import Sequential
# LSTM - Part of RNN
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[52]:


covid


# In[53]:


data = covid[covid['Country/Region'] == 'India']


# In[54]:


data = data.loc[:, ['ObservationDate', 'Confirmed']]


# In[78]:


data


# In[79]:


data = data.groupby('ObservationDate')[['Confirmed']].max().reset_index()


# In[57]:


data


# In[58]:


dataset = data.drop('ObservationDate', axis = 1)


# In[60]:


dataset


# In[61]:


data = np.array(dataset).reshape(-1,1)


# In[62]:


train_data = dataset[:len(dataset)-5]


# In[63]:


test_data = dataset[len(dataset)-5:]


# In[65]:


#MinMaxScaling for Preprocessing data
#scaling large samples into equal ranges of (0,1)
#scaling the dataset
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[66]:


n_input = 5
n_feature = 1


# In[67]:


generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length = n_input, batch_size=10)  #batch size in no. of input samples provided


# In[69]:


#X input value
generator


# In[72]:


## LSTM Model
lstm_model = Sequential()
#layer 1 
lstm_model.add(LSTM(units = 64, return_sequences = True, input_shape = (n_input, n_feature)))
lstm_model.add(Dropout(0.2))
#Hidden Layer 1
lstm_model.add(LSTM(units = 64, return_sequences = True))
lstm_model.add(Dropout(0.2))
#Hidden Layer 2
lstm_model.add(LSTM(units = 64))
lstm_model.add(Dropout(0.2))
#Output Layer
lstm_model.add(Dense(units = 1))

#Optimization and Loss Function
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[73]:


lstm_model.summary()


# In[74]:


#Total Iterations for Training
lstm_model.fit(generator, epochs = 53)


# In[75]:


lstm_model.history.history


# In[76]:


pd.DataFrame(lstm_model.history.history).plot(title="loss vs epochs curve", color='orange', figsize=(12,8))


# In[77]:


lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_feature))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled))
np.round(prediction)


# In[ ]:





# Prediction of the bitcoin price for the next day
# Prediction to d+1

"""

    In order for this algorithm to work, your python environment needs to have the following libraries :
        - tensorflow
        - sklearn
        - matplotlib
        - numpy
        - pandas


"""

####################################################################
"""
                Summary and Explanations

    This algorithm aims to predict the Bitcoin price for the next day based on the last 60 values (2 months).

    The algorithm mainly uses the tensorflow library made by Google.
    I used RNN for building this model since we need the result from the preivous days to estimate the next

    This type of Neural Network is used in almost all of the AI based algorithm that deal with Time-Series

    There is three parts in this algorithm :
        - Data processing
        - AI learning program (using tensorflow)
        - Analysing & plotting the results


    IMPORTANT : 
    By default, the saved model will be used, if you want to compile your own model, you must:
        - Uncomment line : 172
        - Change the string in the : tf.keras.models.load_model('btcAI90%') by changing 'btcAI90%' to 'btcAI', line : 175

"""

####################################################################
# Importing the librairies
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# Reading data and adding a column to have an index
dataBtc = pd.read_csv('data_btc.csv', header=0)
dataBTC = pd.read_csv('price_btc_1h.csv', header=0, sep=',')
dataBitstamp = pd.read_csv('Bitstamp_BTCUSD_d.csv', header=0, sep=',')
# This dataset has a column of Timstamp
dataBitstamp = dataBitstamp.drop(['Unix Timestamp'], axis=1)
print(dataBitstamp.head())

# Data processing and slicing dataframe in order to have the right dimension to input in the RNN
data = pd.DataFrame(dataBitstamp, index=[i for i in range(dataBitstamp.shape[0])])
# The datafiles I had were reversed, meaning that the index of the lastest data was 0, so I had to reverse the dataframe
data = data[::-1]
data = data.drop(['Date','Symbol', 'Volume USD'], axis=1)
print(data.shape)
# RSI integration to the dataset

def RSI(data): 
    dataLength = len(data)
    priceUp = []  
    priceDown = []
    k = 0
    while k < dataLength:
        if k == 0:
            priceUp.append(0)
            priceDown.append(0)
        else:
            diff = data['Close'][k] - data['Close'][k-1]
            if diff > 0:
                priceUp.append(diff)
                priceDown.append(0)
            elif diff <= 0:
                priceDown.append(diff)
                priceUp.append(0)
        k+=1
    avgGain = []
    avgLoss = []
    i = 0
    while i < dataLength:
        if i < 15:
            avgGain.append(0)
            avgLoss.append(0)
        else:
            avgGain.append(sum([priceUp[i-l] for l in range(0, 15)]) / 14)
            avgLoss.append(np.abs(sum([priceDown[i-l] for l in range(0, 15)]) / 14))
        i += 1
    RS = []
    RSI = []
    m = 0
    while m < dataLength:
        if avgGain[m] == 0 and avgLoss[m] == 0:
            RS_TempValue = 0
        else: 
            RS_TempValue = avgGain[m]/avgLoss[m]
        RS.append(RS_TempValue)
        RSI_TempValue = 100 - (100 / (1 + RS_TempValue))
        RSI.append(RSI_TempValue)
        m += 1
    data['RSI'] = RSI
    return data
print(RSI(data))
print('Shape : ',data.shape)
# Splitting the data
index = data.shape[0] - np.floor(data.shape[0] * 0.7)
data_train, data_test = data.loc[:index, :], data.loc[index + 1:, :]

# Need the minimum for the unscaling process that happens later
min = np.min(data_test.iloc[:,3])
print(min)

# Scaling the data in order to facilitate the calculations
scaler = MinMaxScaler(feature_range=(0,1))
data_training, data_testing = scaler.fit_transform(data_train), scaler.fit_transform(data_test)

## Getting the data for the train and test part
X_train = []
y_train = []

def data_selection(data_to_select):
    X = []
    y = []
    for i in range(data_to_select.shape[0] - 60):
        subset = []
        y.append(data_to_select[i + 60, 3])
        for j in range(60):
            subset.append(data_to_select[i + j])
        X.append(subset)
    X,y = np.array(X),np.array(y)

    return X, y

#X_train, y_train = data_selection(data_training)
#X_test, y_test = data_selection(data_test)

X_test = []
y_test = []
X_train = []
y_train = []

# Here I am setting up a moving index, meaning that I make sublists of 60 values
#           For example the first 60 values [0, 59] are predicting the 60th
#               Then the [1, 60] -> 61th
#                        [2, 61] -> 62th
#           And so on, and so forth until I reach the the end of the dataset
# In the end I have len(dataset) - 60 lists of 60 values
# I assign to each of the sublist the next value (the one that I want the algorithm to predict)
#               x0 = [0, 59] | y0 = 61
#               x1 = [1, 60] | y1 = 62

for i in range(60, data_train.shape[0] - 1):
    X_train.append(data_training[i - 60:i])
    y_train.append(data_training[i + 1, 3])

for i in range(60, data_test.shape[0] - 1):
    X_test.append(data_testing[i-60:i])
    y_test.append(data_testing[i + 1, 3])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

def AI_model():
    model = tf.keras.Sequential()

    # Here you might want to change the number of units per layers, because my algorithm ran for 25 min until the model was compiled
    # you can also change the number of epoch, to have a better render time.

    model.add(tf.keras.layers.LSTM(units = 3600, activation=tf.nn.relu, return_sequences=True, input_shape= (X_train.shape[1], 6)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=3600, activation=tf.nn.relu, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=3600, activation=tf.nn.relu, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=1800, activation=tf.nn.relu, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.LSTM(units=900, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))


    # As this is a regression model, we want to calulate the price of the asset for the next day, we use the mean sqaured error as a loss funtion.
    # The adam optimizer is the standard in the AI libraries.
    model.compile(optimizer='adam',
                  loss = 'mean_squared_error', metrics=['accuracy'])

     # Show how the RNN is structured and how many parameters there is.
    print(model.summary())

    # Fitting the model to the train dataset
    # Changing the epoch number will decrease the render time
    model.fit(X_train, y_train, epochs=10, batch_size=32)


    # Saving the model, it last 25 min so I did not want to do it again
    model.save('btcAI')



AI_model()

# Loading the model from the file in the same repository
new_model = tf.keras.models.load_model('btcAI')

# Getting the results on the test dataset
y_preds = new_model.predict(X_test)

# Unscaling the data, in order to have the real results
# This step can be a source of error because we might loose some information by unscaling the data and deforming it
scale = scaler.scale_[3]

y_preds = y_preds/scale + min
y_test = y_test/scale + min


# Printing the latest value for the predicted value and the real value
print("Last real value : ",y_test[-1])
print("Last predicted value : ",y_preds[-1])

####################################################################
"""
        Conclusion

    We can see that the graph is following quite nicely the real data, however there is some delay
    and some gap between the predicted values and the real values. This is mainly due to the great volatility of the market

    Keep in mind that every value of the predicted values list are being predicted using the last 60 and predicting the 61st based
    on the 60 others. So this is not that bad, eventhough the accuracy metrics is not that good.

    In order to improve the model and get closer result we could add columns to the data frame like if the market went up or down by putting 0 or 1,
    Or parse data from twitter, or financial blogs, so that we can give an insight to the algorithm of the mindset of the trader and the public opinion.

"""
####################################################################

"""
Plotting the graph, to visualize the data
"""
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real BTC Price')
plt.plot(y_preds, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction (one day = one value)')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()



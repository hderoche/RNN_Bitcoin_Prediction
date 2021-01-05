# Reccurent Neural Network for Bitcoin Price predictions

**Please use the RNN.ipynb file, where all the analysis is made**


## How it works

I am setting up a moving index, meaning that I make sublists of 60 values
For example the first 60 values : [0, 59] are predicting the 60th
Then the

         [1, 60] -> 61th

         [2, 61] -> 62th
         
And so on, and so forth until I reach the the end of the dataset
In the end I have len(dataset) - 60 lists of 60 values
I assign to each of the sublist the next value (the one that I want the algorithm to predict)

               x0 = [0, 59] | y0 = 61

               x1 = [1, 60] | y1 = 62

In the end, by taking the last 30 days I can predict the next day
This also works with the data of BTC by hours

### Incorporate basic indicators
I will then try to add some of the most basic indicator to the dataset so that I might have some better price predicitons.

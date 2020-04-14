# Analysis-and-prediction-of-SPY-500

Deep Learning Architecture:
I have used deep learning approach for predictions of the dataset. Although, orthodox approaches like ARIMA and SARIMAX methods could be used, but there are additional 3 parameters (p, d, q) where p is the number of autoregressive terms, d is the number of nonseasonal differences needed for stationarity, and q is the number of lagged forecast errors in the prediction equation. Searching for these 3 optimal values requires an extensive search, not to mention the increased computation cost, and the accuracy of predictions depends highly on these parameters. When using a seasonal ARIMA model (SARIMAX), 3 additional terms are introduced which further increase the computation. On the contrary, with the same computational cost, LSTM’s deep learning network provides far better accuracy than ARIMA models. Also, LSTM’s have an explicit control on which information to forget and which information to pass forward, known as long term dependency which is briefly explained below. Also, deploying the network is very easy and multiple architectures could be used for multiple technical indicators.
Long Short Term Memory cells are like mini neural networks designed to allow for memory in a larger neural network. This is achieved through the use of a recurrent node inside the LSTM cell. This node has an edge looping back on itself with a weight of one, meaning at every feedforward iteration the cell can hold onto information from the previous step, as well as all previous steps. Since the looping connection’s weight is one, old memories won’t fade over time like they would in traditional RNNs.
LTSMs and recurrent neural networks are as a result good at working with time series data thanks to their ability to remember the past. By storing some of the old state in these recurrent nodes, RNNs and LSTMs can reason about current information as well as information the network had seen one, ten or a thousand steps ago. Even better, I don’t have to write my own implementation of an LSTM cell.

The Dataset:
The good thing about stock price history is that it’s basically a well labelled pre-formed dataset. I knew one of the famous api to use i,e, yahoo finance and used it to fetch data directly from the server from 29th Jan, 1993 up till 9th April 2020, one of the data collection methods mentioned in the task details. They offered the daily price history of NASDAQ stocks for the past 27 years. This included the open, high, low, close and volume of trades for each day, from today all the way back up to 1999. I opted to download the datasets and save them in CSV format so I could use them as often as I wanted. For the stocks that had their IPO listing within the past 27 years, the first day of trading that stock often looked anomalous due to the massively high volume. This inflated max volume value also affected how other volume values in the dataset were scaled when normalising the data, so I opted to drop the oldest data points out of every set. I also drop the date since the model doesn’t need to know anything about when trades happened, all it needs is well ordered time series data. However, while laying out visualizations and predictions, dates are used for proper interpretation.
I also keep track of the number of {history_points} we want to use; the number of days of stock history the model gets to base its predictions off of. So, if history_points is set to 60, the model will train on and require the past 60 days of stock history to make a prediction about just the next day.
In order to scale the data, I have used MinMax scaler from sklearn which scales the whole data between 0 and 1.
The ohlcv_histories list will be our x parameter when training the neural network. Each value in the list is a numpy array containing 60 open, high, low, close, volume values, going from oldest to newest. This is controlled by the history_points parameter, as seen inside the slice operation.
So for each x value we are getting the [i: i + history_points] stock prices (remember that numpy slicing is [inclusive:exclusive]), then the y value must be the singular [i + history_points] stock price; the stock price for the very next day.
Here we also have to choose what value we are intending on predicting. I decided to predict the open value for the next day, so we need to get the 0-th element of every ohlcv value in the data. There’s also a variable called y_normaliser to hold on to. This is used at the end of a prediction, where the model will spit out a normalised number between 0 and 1, we want to apply the reverse of the dataset normalisation to scale it back up to real world values. In the future we will also use this to compute the real world (un-normalised) error of our model. Then to get the data working with Keras I make the y array 2-dimensional by way of np.expand_dims(). And finally, I keep hold of the unscaled next day open values for plotting results later. Just before returning the data we check that the number of x’s == the number of y’s.

The Model:
I started this project with keras for deploying the lstm architecture because it is a very good api that runs on top of Tensorflow. It is very good for deploying small scale neural network architectures like this in very less time and the debugging part is also easy.
I’ll go over the most basic model that I came up with first. The basic model architecture image could be found in the image file named "basic_model.png".
This is the basic lstm model that I first used consisted of only a singe channel with shape (history_points, 7), since each input data point is an array shaped like [history_points × OHLCV]. The model has 60 LSTM cells in the first layer, a dropout layer to prevent overfitting and then some dense layers to bring all of the LSTM data together.
An important feature of this network is the linear output activation, allowing the model to tune its penultimate weights accurately. I got a final evaluation score of 0.00019, which seems super low but remember that this is the mean squared error of the normalised data. After scaling this value will go up significantly, so it’s not a great metric for loss.
The Evaluation
To more accurately evaluate the model, let’s see how it predicts the test set in comparison with the real values. First, we scale the predicted values up, then we compute the mean squared error, but then to make the error relative to the dataset we divide it by the ‘spread’ of the test data — the max test data point minus the min.
This gives us an adjusted mean squared error of 20.17. Is that good? It’s not amazing, it means on average the predicted line deviates over 20% from the actual. Now, that’s not a very good mse score. Let’s go for feature engineering and a more complex lstm model.

The Improvements:
We could try to make our model more complex, and also increase the size of the dataset. Let’s start with trying to create a more complex model.
A common metric used by stock market analysts are technical indicators. Technical indicators are math operations done on stock price history, and are traditionally used as visual aids to help identify the direction the market is going to change in. We can augment our model to accept these technical indicators through a secondary input branch.
For now, let’s use the Simple Moving Average (SMA) and the Moving Average Convergence Divergence (MACD) indicator as an extra input into our network. 
A simple moving average, the estimate of the trend-cycle at time t, is obtained by averaging values of the time series within k periods of t. Observations that are nearby in time are also likely to be close in value. Therefore, the average eliminates some of the randomness in the data, leaving a smooth trend-cycle component. We call this an m - MA, meaning a moving average of order m. I have used m = 3.
The MACD is calculated by subtracting the 26-period Exponential Moving Average from the 12-period EMA with the following formulae:
EMA = Price(t)*k + EMA(y)*(1-k) where:
t = today
y = yesterday
N = number of days in EMA
k = 2/(N+1)

Feature Engineering:
Apart from technical indicators, 2 new features have been used for developing a better unbiased model.
Percentage change of High and Low values – This shows by what percent do the high and low stock values change w.r.t. close values. 
It is calculated as:
             HL_PCT =   (High-Low)/Close*100
Percentage Change of Open and Close Values – This shows by what percent do the Open and Close stock values changes. 
It is calculated as:
     PCT_change = (CLose-Open)/Open*100
     
Improved Model Architecture:
Note how we used technical_indicators.shape as the input shape for the tech_input layer. This means that any new technical indicators we add will fit in just fine when we recompile the model.
The evaluation code has to be changed to match this dataset change as well.
We pass in a list of [ohlcv, technical_indicators] as the input to our model. This order matches the way we defined our model’s input.
The improved model architecture model can be fould in "improved_model.png"
And we get an adjusted mean squared error of 4.99! Much lower, and the prediction appears to fit significantly closer to the test set when plotted.
Plots can be seen in the image folder named "plots".
Some of the plots are made with bokeh and plotly and can be best visualized inside the notebook, once it is executed.
At early stage, the validation loss is very different than training loss. This is due to the fact that at the start, the model is learning the time dependent features and trying to form long term dependencies but as the epochs increase, both the losses stabilize gradually because by then, the model has eventually learned the long-term dependencies as well as got the ability to differentiate between noise and data.

Logical Results and Actionable Trading Strategy:
The various stock values plots of S&P 500 are plotted in the jupyter notebook provided. The plotting is done with bokeh and the output is embedded in the notebook. Here is a small glimpse of the plot. An extensive view can be obtained in the notebook itself.

We see that whenever the dark blue line(200 day MA) crosses under the light blue line(50 day MA), one should sell and buy when the dark blue line(200 day MA) crosses over the the light blue line(50 day MA). Fore more succint information, get your asses on the file "trading strategy.png" inside the plots folder.

Actionable insights:
•	There are these seasonal dips that which happen every time in the stock, probably due to some cyclical components.
•	As we go towards the recent data, the dips are more prominent that previous dips, meaning the uncertainty in the stock has recently increased.
•	In the open and close plot, it can be seen that whenever the stock in recovering from the dip, the open and close values differ greatly.
•	The return of the stock is the difference between the open value and the previous date closing value. This return is very moderate because it goes as high as 0.15 and as low as -0.10, which means, on an average, there is not much difference between the opening value and the previous day closing value of the stock.

Creative ideas for future data and model building
•	We can also experiment with using a larger dataset. If more time would have been provided, we could train on intraday dataset, thus increasing the size of the dataset.
•	A more deeper LSTM network could be used for prediction, with larger epochs to get better accuracy. But as the time given was not much and also, I don’t have any gpu’s, it is impossible to implement such a deeper network.
•	Additional technical indicators like sentiment analysis of various news related to the stocks could be also as a good feature for training the model.
•	I’d also like to look into giving the model more data by having more LSTM branches, one for each timestep available on Yahoo finance, so the network can make decisions based on short, medium- and long-term trends.
On a broader note, namely, the technical indicators used, history_points hyperparameter and model architecture are all things that I would like to optimise in the future.




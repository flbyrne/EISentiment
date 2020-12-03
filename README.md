# EISentiment

## Get Trading Signals before Unemployment Announcements

This tool allows investors to profit from stock price changes when unemployment rates are announced. It issues buy or sell signals immediately after the market close on the day before the unemployment announcement using machine learning predictions. Conventional wisdom would advice to buy if rates are predicted to go down and sell if rates are predicted to go up. EISentiment beats conventional wisdom by predicting the stock direction based on the news sentiment and the forecasted unemployment rate.

EISentiment uses sentiment from stock news headlines, a time series price estimator and predicted unemployment rates as features for a regression machine learning model for each stock. Out of the stocks in the Standard and Poor 500 Index, the stocks selected were the ones that had sufficient news to train the models. The type of regressor selected for each stock was based on its training and test performance using accuracy and mean absolute percentage error (MAPE) and those are either Random Forest, Ridge Regression or Linear Regression, depending on the stock. I chose regression models over classifiers because they provided a better accuracy on the final prediction, therefore their results are further processed by a simple classifier to determine the signal direction. The news sentiment classifier is a Stochastic Gradient Descent (SGD) model trained on three years of news headlines and uses NLP to process the news text. The time series predictions are made using an ARIMA model trained with ten years of price data. Finally, the predicted unemployment rates are sourced from a financial website. For information about data sources see data sources.

Try it here: https://eisentiment.herokuapp.com!


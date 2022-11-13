How to launch stock prediction model:

install pandas
install pandas-datareader
install matplotlib
install keras
install sklearn
install scikit-learn
install tensorflow

Open main.py file.
Choose company stock you want to predict. Using company short stock name, database is from yahoo.

Optional:
Choose prediction days window on how many days model will make predictions (default is 60)
Choose epochs count on how long models will train (default is 100).

Script is using LSTM and GRU models to predict day ahead price of the selected stock.
Data is saved to logs folder.
Use visual graphs to judge model performance.
Next day Close price predictions are saved in stock_predictions.txt file.

Chose not to save pretrained model as it is better to train and test with new stock data.
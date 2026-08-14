# AAPL Stock Price Prediction using Pytorch and LSTM

## About the Project

In this project, I used historical **Apple (AAPL)** stock data to predict the next closing price using an LSTM neural network.

The model learns from previous stock prices and uses them to predict the next price.

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* yfinance
* Scikit-learn

## Project Workflow

1. Download historical AAPL stock data using `yfinance`
2. Select the closing price
3. Scale the data using `StandardScaler`
4. Create time-series sequences
5. Split the data into training and testing sets
6. Build a 2-layer LSTM model
7. Train the model using MSE Loss and Adam optimizer
8. Evaluate the model using RMSE
9. Visualize actual vs predicted prices

## 🧠 Model

* Input size: `1`
* Hidden size: `64`
* LSTM layers: `2`
* Output size: `1`
* Loss function: `MSELoss`
* Optimizer: `Adam`
* Epochs: `200`

The model uses the previous **29 days of closing prices** to predict the next closing price.

## 📊 Results

| Metric     |  Value |
| ---------- | -----: |
| Train RMSE | 3.2433 |
| Test RMSE  | 5.0732 |

The project also includes a visualization comparing the **actual and predicted AAPL prices** and the prediction error.

---

Currently, the model uses only historical closing prices.
This project is created for **learning and educational purposes**. It is not intended for financial decision-making or investment advice.


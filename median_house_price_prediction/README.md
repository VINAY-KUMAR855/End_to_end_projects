# California Housing Price Prediction

In this project I learned few concept’s.
1. Before moving to model training we need to understand problem and data, what each feature tells and explore dataset.
![Working](images/california_housing_prices_plot.png)
2. Most important is feature engineering. Is it possible to create new features from existing one?. At the end i got Out of top 10 feature_importance, 8 are from feature engineering.
![Working](images/random_forest_feature_importance.png)
3. Check whether the data is skewed or not. Because model excepts uniform data. I used log transformation.
![Working](images/population_log_transformation.png)
4. I Use a pipelines for prevent data leakage. Also I learned how to build custom transformers.
5. Finally train model and evaluate. Experiment with different models and hyper parameters.
6. I wrote backend using fastapi. After all I dockerize everything.

This project gives the basic understanding of core machine learning.

## How to Run with Docker

### 1. Build the Docker Image
```bash
cd backend
docker build -t housing-api .
```

### 2. Run the Docker Container
```bash
docker run -p 8000:8000 housing-api
```

### 3. Open API Documentation
* http://127.0.0.1:8000/docs

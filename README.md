# Wine Price Prediction

Those models have to predict the price of a wine bottle based on a collection of over hundred of thousands of wine reviews and other relative features. The dataset has been taken from the challenge ["Business Game 2018"](http://www.bee-viva.com/competitions/career_2018).

| Model                                  | train set RMSE | test set RMSE |
|----------------------------------------|----------------|---------------|
| Bert regressor                         | 12.67          |               |
| Bert classifier (label discretization) | 16.05          |               |
| Xgboost + Grid search                  | 8.11           | 20.78         |
| Random forest + Grid search            | 7.85           | 22.56         |
| Ensemble (Xgboost + Random forest)     | 7.16           | 20.50         |

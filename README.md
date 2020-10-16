# Wine Price Prediction

Those models have to predict the price of a wine bottle based on a collection of over hundred of thousands of wine reviews and other relative features. The dataset has been taken from the challenge ["Business Game 2018"](https://medium.com/genifyai/banking-products-recommendation-engine-what-we-learnt-building-our-minimum-viable-product-mvp-7097a52bb413).

| model                                  | train set RMSE | test set RMSE |
|----------------------------------------|----------------|---------------|
| Bert regressor                         | 12.67          |               |
| Bert classifier (label discretization) | 16.05          |               |
| Xgboost + Grid search                  | 8.11           |               |
| Random forest + Grid search            | 7.85           |               |
| Ensemble (Xgboost + Random forest)     | 7.16           |               |

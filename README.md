# House_Price_Prediction

The project 'House Price Prediction using Machine Learning' demonstrated the efficacy of various machine learning algorithms for predicting house prices. The Decision Tree Regressor model outperformed all others.

## Approach
<ul>
<li><strong>Data Collection:</strong> The dataset is collected and loaded into a pandas DataFrame.</li>
<li><strong>Data Preprocessing:</strong> The data is cleaned and preprocessed. This includes handling missing values, outliers, and inconsistencies. Feature engineering is performed to extract relevant features from the data that might impact house prices.</li>
<li><strong>Model Building:</strong> Several machine learning algorithms are implemented including Support Vector Regressor (SVR), Random Forest Regressor, Decision Tree Regressor, Gradient Boosting Regressor, and K-Nearest Neighbors (KNN). These models are trained and tested on a split of the dataset to predict house prices based on historical patterns.</li>
<li><strong>Model Evaluation:</strong> The model’s performance is assessed using the Mean Absolute Percentage Error (MAPE) as the evaluation metric.</li>
<li><strong>Deployment:</strong> The final model is deployed to make real-time prediction</li>
</ul>

## Algorithm Selection
<ul>
<li><strong>Random Forest Regressor: </strong>This is a meta estimator that fits several classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.</li>
<li><strong>Decision Tree Regressor:</strong> This algorithm builds regression models in the form of a tree structure and breaks down our dataset into smaller subsets while at the same time an associated decision tree is incrementally developed.</li>
<li><strong>Gradient Boosting Regressor:</strong> This algorithm produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.</li>
<li><strong>K-Nearest Neighbors (KNN):</strong> This algorithm assumes that similar things exist in close proximity and predicts the value of any given point in the dataset by averaging the values of the ‘k’ closest points.</li>
</ul>

## Model Performance: 


- **Random Forest Regressor:** The Random Forest model achieved a MAPE of **19.19%**. This shows that the Random Forest model’s predictions were on average 19.20% different from the actual prices.

- **Decision Tree Regressor:** The Decision Tree model achieved a MAPE of **23.18%**. This means that the Decision Tree model’s predictions were on average 22.96% different from the actual prices.

- **Gradient Boosting Regressor:** The Gradient Boosting model achieved a MAPE of **19.10%**. This indicates that the Gradient Boosting model’s predictions were on average 19.16% different from the actual prices.

- **K-Nearest Neighbors (KNN):** The KNN model achieved a MAPE of **20.56%**. This shows that the KNN model’s predictions were on average 20.58% different from the actual prices.

  ## Result

  <img src="https://github.com/user-attachments/assets/ccb4e734-3f2e-4443-9499-5aeb5aed7441" alt="Model Performance" width="850" height="556">
  
  <img src="https://github.com/user-attachments/assets/ecf6eaee-9ea6-467a-8e5c-a5c14f2cba5b" alt="Model Performance" width="850" height="556">
  
  <img src="https://github.com/user-attachments/assets/af310148-d9e1-420b-b7a8-80685d646fe5" alt="Model Performance" width="850" height="556">

  <img src="https://github.com/user-attachments/assets/431d41ba-05d8-4bb9-8d89-aa59ba34c6ec" alt="Model Performance" width="850" height="386">

# Thank You


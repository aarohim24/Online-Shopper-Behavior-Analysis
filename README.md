**Shopper Behavior Analysis & Purchase Prediction**

This project analyzes synthetic shopper behavior data to identify customer segments using clustering, and predicts purchase likelihood using a Random Forest Classifier. It also includes sentiment analysis of review text and various feature engineering steps to enhance model performance.

Features
* **Synthetic Data Generation** (1000 sessions)
* **Sentiment Analysis** using HuggingFace Transformers
* **Feature Engineering**: cart value, avg. time per page, encoded demographics, and more
* **Customer Segmentation** via KMeans Clustering
* **Purchase Prediction** using Random Forest with Hyperparameter Tuning (GridSearchCV)
* **Interactive Visualizations** using Plotly
* **Model Export** using Joblib for deployment

Technologies Used
* `pandas`, `numpy` - Data manipulation
* `scikit-learn` - Preprocessing, Clustering, Classification, Model Evaluation
* `transformers` - Pre-trained sentiment analysis pipeline
* `matplotlib`, `seaborn`, `plotly` - Data visualization
* `joblib` - Model saving


Data Fields (Synthetic)
* `time_spent` – Time spent on site (in seconds)
* `pages_visited` – Number of pages browsed
* `cart_items`, `add_to_cart` – Cart activity
* `checkout_initiated`, `purchase_made` – Conversion flags
* `review_text` – Synthetic customer reviews
* `user_age`, `user_gender`, `device_type` – Demographics
* `loyalty_program`, `social_media_influence` – Marketing flags
* `abandoned_cart` – Derived field
* `review_sentiment` – NLP-based polarity score

Model Training
* **Clustering**: KMeans (4 clusters)
* **Classification**: Random Forest with grid search for tuning
* **Metrics**: Accuracy, Confusion Matrix, Classification Report

Outputs
* `shopper_analysis_results.csv` – Final dataset with all engineered features and predictions
* `random_forest_model.pkl` – Trained model saved with Joblib

Visualizations
* **3D Shopper Segments**: By time spent, pages visited, and cart value
* **Feature Importance**: Based on Random Forest model
* **Confusion Matrix**: Model evaluation visualization

Future Improvements
* Connect to real-time user tracking data
* Build a web dashboard for business insights
* Incorporate session-based deep learning models

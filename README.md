# Heart Disease Prediction

This project focuses on predicting heart disease using machine learning models, with a particular focus on Logistic Regression.  
The dataset used is the UCI Heart Disease dataset (~300 samples).

---

## 📂 Project Structure
- train.py → Clean Python script for training and saving the final Logistic Regression model.  
- logistic_model.pkl → Trained Logistic Regression model (saved with joblib).  
- notebooks/ → Contains the main Jupyter notebook with experiments and exploratory work.  
- heart-disease.csv → Dataset used for training and evaluation.  

---

## 📊 Conclusion / Model Evaluation

### 1. Models Tried
I experimented with multiple classification models including Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN).  
Among these, Logistic Regression initially showed the best performance, so I decided to focus on improving it further.  

### 2. Data Preprocessing & Feature Engineering
- One-Hot Encoding for categorical features  
- Standard Scaling for numerical features  
- SMOTE oversampling to balance the dataset  
- New feature risk = chol / age, which improved model interpretability  

### 3. Model Comparison

| Metric     | Baseline Model | Improved Model |
|------------|----------------|----------------|
| Accuracy   | 0.844          | 0.848          |
| Precision  | 0.820          | 0.860          |
| Recall     | 0.921          | 0.878          |
| F1-score   | 0.867          | 0.858          |

### 4. Results & Insights
- Improved model shows slightly higher accuracy and precision.  
- Feature risk turned meaningful compared to chol and age individually.  
- SMOTE balanced the dataset, reducing bias towards the majority class.  
- ROC and confusion matrix confirmed more stable classification.  

### 5. Next Steps
- The model was validated using cross-validation, so results are reliable.  
- Future improvements: testing advanced models like XGBoost/LightGBM.  
- Explore more domain-specific features.  
- Collect more data (current dataset is small ~300 samples, which limits accuracy below 95%).  

---

## ⚙️ Requirements
- Python 3.x  
- scikit-learn  
- imbalanced-learn  
- pandas  
- matplotlib  
- joblib  

Install all dependencies:
`bash
pip install -r requirements.txt

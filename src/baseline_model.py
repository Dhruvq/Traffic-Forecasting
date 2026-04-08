import os
import pandas as pd
import numpy as np

class CustomStandardScaler:
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class CustomOneHotEncoder:
    def fit(self, X):
        X = np.asarray(X).astype(str)
        self.categories_ = []
        for i in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, i]))
        return self
        
    def transform(self, X):
        X = np.asarray(X).astype(str)
        out = []
        for i in range(X.shape[1]):
            col_data = np.zeros((X.shape[0], len(self.categories_[i])))
            for j, cat in enumerate(self.categories_[i]):
                # Note: this relies on ignoring unknown categories essentially as 0
                col_data[:, j] = (X[:, i] == cat).astype(float)
            out.append(col_data)
        return np.hstack(out)
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def mean_squared_error(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

class CustomRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def get_params(self, deep=True):
        return {"alpha": self.alpha}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Add column of ones for intercept
        X_design = np.column_stack((np.ones(n_samples), X))
        
        # Regularization matrix (identity, but 0 for intercept)
        I = np.eye(n_features + 1)
        I[0, 0] = 0.0
        
        # Closed-form solution: (X^T X + alpha*I) * theta = X^T y
        A = X_design.T @ X_design + self.alpha * I
        b = X_design.T @ y
        
        # Solve linear system
        theta = np.linalg.solve(A, b)
        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

def main():
    print("Loading preprocessed data...")
    # Load dataset
    df = pd.read_csv('../data/preprocessed_traffic_data.csv')
    
    # We drop date_time as we've already extracted features from it in data preparation
    X = df.drop(columns=['traffic_volume', 'date_time'])
    y = df['traffic_volume']
    
    # Time-series aware split (e.g., 80% train, 20% test sequence)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define preprocessing steps
    numeric_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    categorical_features = ['holiday', 'weather_main', 'hour', 'day_of_week', 'month']
    
    # Process data independently using custom helpers
    print("Applying preprocessing...")
    scaler = CustomStandardScaler()
    ohe = CustomOneHotEncoder()
    
    X_train_num = scaler.fit_transform(X_train[numeric_features])
    X_test_num = scaler.transform(X_test[numeric_features])
    
    X_train_cat = ohe.fit_transform(X_train[categorical_features])
    X_test_cat = ohe.transform(X_test[categorical_features])
    
    # Concatenate numerical and categorical features
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])
    
    print("Training baseline Custom Ridge Regression model...")
    model = CustomRidgeRegression(alpha=1.0)
    model.fit(X_train_processed, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_processed)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("--- Baseline Model Results ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

if __name__ == '__main__':
    # Adjust working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()

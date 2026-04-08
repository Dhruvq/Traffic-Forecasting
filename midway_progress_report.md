# Traffic Volume Forecasting: Midway Progress Report

## 1. Project Overview
The objective of this project is to develop a predictive model for hourly interstate traffic volume (in vehicles per hour) on Interstate 94, using the Metro Interstate Traffic Volume dataset from the UCI Machine Learning Repository. We aim to evaluate multiple models, starting from simple linear models to complex deep learning approaches (FFNN, LSTM), to capture both linear dependencies and complex temporal dynamics.

## 2. Data Exploration and Preprocessing
The dataset contains 48,204 continuous hourly observations of traffic volume, weather conditions, and temporal indicators. 

### Exploratory Data Analysis (EDA)
- **Data Completeness**: Initially, we checked for missing values. Most columns were fully populated. The `holiday` column reported a large number of 'missing' values, which we accurately identified as non-holiday days.
- **Traffic Volume Distribution**: The target variable ranges from 0 to over 7,000 vehicles per hour. The distribution is bimodal, indicating distinct high-traffic and low-traffic periods.
- **Temporal Patterns**: Visualizing average traffic by the hour of the day revealed clear daily seasonality. Peaks typically occur during the morning rush hour (~7 AM) and evening rush hour (~4 PM - 5 PM), confirming the strong temporal dependence of the data. 

*(EDA scripts and generated plots are available in the `src/` and `images/` directories respectively.)*

### Preprocessing Steps
To prepare the dataset for our baseline model, we implemented the following pipeline:
1. **Handling Missing Values**: Filled missing values in the `holiday` feature with the string literal 'None'.
2. **Temporal Feature Engineering**: Converted the `date_time` column to a datetime format and extracted `hour`, `day_of_week`, and `month` as categorical features to capture cyclicity.
3. **Data Splitting**: We performed a time-series aware train/test split. The dataset was sorted chronologically, and the first 80% was used for training (38,563 samples), while the remaining 20% was set aside for testing (9,641 samples).
4. **Encoding and Scaling**: 
   - `StandardScaler` was applied to continuous features (`temp`, `rain_1h`, `snow_1h`, `clouds_all`).
   - `OneHotEncoder` was applied to categorical/temporal features (`holiday`, `weather_main`, `hour`, `day_of_week`, `month`).
5. **Feature Filtering**: We dropped detailed textual descriptions such as `weather_description` in favor of the broader `weather_main` category to prevent feature explosion in the baseline model.

## 3. Initial Baseline Model Results
For our baseline, we trained a **Ridge Regression** model. This provides a strong linear benchmark with L2 regularization to prevent overfitting on the categorical feature space.

**Evaluation Results (Test Set)**:
- **MSE (Mean Squared Error)**: 675,980.50
- **RMSE (Root Mean Squared Error)**: 822.18
- **MAE (Mean Absolute Error)**: 595.93

**Interpretation**: Given that traffic volume ranges from 0 to over 7,000, an MAE of ~596 vehicles per hour is a reasonably solid starting point for a linear model. However, it indicates room for improvement, likely because Ridge Regression cannot capture the complex, non-linear relationships and sequential dependencies inherent in the traffic data.

## 4. Challenges Faced and Next Steps

**Challenges Faced**:
1. **Dataset Delivery Changes**: The dataset was not accessible through the standard `ucimlrepo` API due to formatting choices in the source repository. This required us to write a custom extraction script that pulls the dataset zip directly and decompresses the `csv.gz` file on the fly.
2. **Time-Series Split Leakage**: When working with standard scikit-learn random splits, we have to be extremely careful to avoid data leakage (using future data to predict the past). Implementing chronological splitting effectively mitigates this.

**How We Plan to Address Them (Next Steps)**:
- **Implementation of FFNN**: We will start building the Feedforward Neural Network (Multi-Layer Perceptron), experimenting with non-linear activation functions (ReLU), and applying dropout and L2 weight decay to map out the non-linear boundaries.
- **Sequence Processing for LSTM**: The immediate next technical challenge is to reshape our current 2D tabular data into 3D sequences (samples, timesteps, features) using sliding windows, which is required for training the Long Short-Term Memory (LSTM) network.
- **Hyperparameter Tuning**: Once all pipelines are functional, we will orchestrate a grid/random search for hyperparameter optimization to yield the best-performing models to compare against our Ridge baseline.

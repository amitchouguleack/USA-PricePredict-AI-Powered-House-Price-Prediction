import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
sns.set(color_codes=True)

df = pd.read_csv('USA_Housing.csv')
df.head(5)

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
plt.title("Missing Data: Training Set")
plt.show()

df.count()

print(df.duplicated().sum())
df.dropna(inplace=True)

df.drop('Address', axis=1, inplace=True)
df.head(5)

# Renaming the columns for better understanding
df = df.rename(columns={
    'Avg. Area Income': 'Avg_Area_Income',
    'Avg. Area House Age': 'Avg_Area_House_Age',
    'Avg. Area Number of Rooms': 'Avg_Area_Number_of_Rooms',
    'Avg. Area Number of Bedrooms': 'Avg_Area_Number_of_Bedrooms',
    'Area Population': 'Area_Population'
})

df.head(5)

# Setting the target variables
x = df.drop('Price', axis=1, inplace=False)
y = df['Price']

# Preprocessing the X Variable
pre_processor = preprocessing.StandardScaler().fit(x)
x_transform = pre_processor.fit_transform(x)

x_transform.shape
x_transform

# Preprocessing the Y variable
pre_processor_y = preprocessing.StandardScaler().fit(y.values.reshape(-1, 1))
y_transform = pre_processor_y.fit_transform(y.values.reshape(-1, 1))

y_transform.shape
y_transform


# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_transform, y_transform, test_size=0.3, random_state=101)

# Arranging the models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()
gradient_model = GradientBoostingRegressor()

# Fitting the models - Linear
linear_model.fit(x_train, y_train)
# Fitting the models - Decision Tree
tree_model.fit(x_train, y_train)
# Fitting the models - Random Forest
forest_model.fit(x_train, y_train)
# Fitting the models - Gradient Boosting
gradient_model.fit(x_train, y_train)

# Predicting the output on test data from all the respective models
y_pred_linear = linear_model.predict(x_test)
y_pred_tree = tree_model.predict(x_test)
y_pred_forest = forest_model.predict(x_test)
y_pred_gradient = gradient_model.predict(x_test)

# ScatterPlot for visualizing the data after transformation
for y_pred in [y_pred_linear, y_pred_tree, y_pred_forest, y_pred_gradient]:
  # Ensure y_test and y_pred are 1-dimensional
  y_test_1d = y_test.flatten() if y_test.ndim > 1 else y_test
  y_pred_1d = y_pred.flatten() if y_pred.ndim > 1 else y_pred
  sns.scatterplot(x=y_test_1d, y=y_pred_1d, color='blue', label='Actual Data Points')
  plt.plot([min(y_test_1d), max(y_test_1d)], [min(y_test_1d), max(y_test_1d)], color='red', label="Ideal Line")
  plt.legend()
  plt.show()

# Metrics Calculation

#========Mean Square Error=================#
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mse_forest = mean_squared_error(y_test, y_pred_forest)
mse_gradient = mean_squared_error(y_test, y_pred_gradient)

#========R2 Value=================#
r2_linear = r2_score(y_test, y_pred_linear)
r2_tree = r2_score(y_test, y_pred_tree)
r2_forest = r2_score(y_test, y_pred_forest)
r2_gradient = r2_score(y_test, y_pred_gradient)

#========Mean Absolute Error=================#
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mae_gradient = mean_absolute_error(y_test, y_pred_gradient)

#========Root Mean Square Error=================#
rmse_linear = np.sqrt(mse_linear)
rmse_tree = np.sqrt(mse_tree)
rmse_forest = np.sqrt(mse_forest)
rmse_gradient = np.sqrt(mse_gradient)

# Printing the results for Linear Regressor
print("Linear Regressor")
print("==" * 30)
print(f"MSE: {mse_linear}")
print(f"R2: {r2_linear}")
print(f"MAE: {mae_linear}")
print(f"RMSE: {rmse_linear}")
print("==" * 30)

print("\n")

# Printing the results for Decision Tree Regressor
print("Decision Tree Regressor")
print("==" * 30)
print(f"MSE: {mse_tree}")
print(f"R2: {r2_tree}")
print(f"MAE: {mae_tree}")
print(f"RMSE: {rmse_tree}")
print("==" * 30)

print("\n")

# Printing the results for Random Forest Regressor
print("Random Forest Regressor")
print("==" * 30)
print(f"MSE: {mse_forest}")
print(f"R2: {r2_forest}")
print(f"MAE: {mae_forest}")
print(f"RMSE: {rmse_forest}")
print("==" * 30)

print("\n")

# Printing the results for Gradient Boosting Regressor
print("Gradient Boosting Regressor")
print("==" * 30)
print(f"MSE: {mse_gradient}")
print(f"R2: {r2_gradient}")
print(f"MAE: {mae_gradient}")
print(f"RMSE: {rmse_gradient}")
print("==" * 30)

# Distribution plot to check if the result is somewhat different from the non transformed data or not
for y_pred in [y_pred_linear, y_pred_tree, y_pred_forest, y_pred_gradient]:
  residual = y_test - y_pred
  sns.distplot(residual, kde=True)
  plt.title("Distribution Plot for Residual Values")
  plt.show()

# Comparison of results according to mean square error
model_mse_scores = {
    "Linear Regression" : mse_linear,
    "Decision Tree" : mse_tree,
    "Random Forest" : mse_forest,
    "Gradient Boosting Regressor" : mse_gradient
}

# Sort the model scores in ascending order based on their values (lower values first)
sorted_scores = sorted(model_mse_scores.items(), key=lambda x: x[1])

# Display the ranking of models
print("Model Ranking - (Lower Values are better)")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")

# Comparison of results according to r2 value
model_r2_scores = {
    "Linear Regression" : r2_linear,
    "Decision Tree" : r2_tree,
    "Random Forest" : r2_forest,
    "Gradient Boosting Regressor" : r2_gradient
}

# Sort the model scores in ascending order based on their values (lower values first)
sorted_scores = sorted(model_r2_scores.items(), key=lambda x: x[1])

# Display the ranking of models
print("Model Ranking - (Higher Values are better)")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")
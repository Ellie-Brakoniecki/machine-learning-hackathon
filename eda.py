#%% 
import pandas as pd
ames = pd.read_csv("ames.csv")

# %%
ames 
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ames = pd.read_csv("ames.csv")

print(ames.head())

print("Dataset dimensions:", ames.shape)

print(ames.info())

print(ames.describe())



#%%
plt.figure(figsize=(10,10))


corr = ames.select_dtypes('number').corr()
sns.heatmap(corr, cmap="cool")
#%%
numerical = ames.select_dtypes(include=["int64", "float64"]).columns

def correlation_plots(target_variable):
    for column in numerical:
        if column != target_variable: 
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=ames, x=column, y=target_variable)
            plt.title(f"Correlation between {column} and {target_variable}")
            plt.xlabel(column)
            plt.ylabel(target_variable)
            plt.show()


independent_variables = ["Year Remod/Add", "Overall Qual", "SalePrice", "Garage Cars", "Full Bath", "Total Bsmt SF", "Enclosed Porch"]
target_variable = "Year Built"




# %%
correlation_plots("SalePrice")
correlation_plots("Year Built")
# %%
ames_selected = ames[independent_variables + [target_variable]].copy()

for column in ames_selected.columns:
    if ames_selected[column].dtype == "object":
        ames_selected[column].fillna(ames_selected[column].mode()[0], inplace=True)
    else:
        ames_selected[column].fillna(ames_selected[column].median(), inplace=True)

missing_values_cleaned = ames_selected.isnull().sum()

missing_values_cleaned

X = ames_selected[independent_variables]
y = ames_selected[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
# Baseline model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression Performance:")
print("Mean Squared Error (MSE):", mse_lr)
print("R^2 Score:", r2_lr)
# %%
from sklearn.ensemble import RandomForestRegressor
#Random forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest Performance:")
print("Mean Squared Error (MSE):", mse_rf)
print("R^2 Score:", r2_rf)

# %%
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print("Gradient Boosting Performance:")
print("Mean Squared Error (MSE):", mse_gb)
print("R^2 Score:", r2_gb)

# %%
feature_importances_rf = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances_gb = pd.Series(gb_model.feature_importances_, index=X_train.columns)

# %%
feature_importances_rf
# %%
feature_importances_gb
# %%
import matplotlib.pyplot as plt

# feature importance for rf
plt.figure(figsize=(12, 6))
plt.title("Feature Importance - Random Forest")
plt.barh(range(len(feature_importances_rf)), feature_importances_rf, align="center")
plt.yticks(range(len(feature_importances_rf)), feature_importances_rf.index)
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()  
plt.show()

# Plotting feature importance for gb
plt.figure(figsize=(12, 6))
plt.title("Feature Importance - Gradient Boosting")
plt.barh(range(len(feature_importances_gb)), feature_importances_gb, align="center")
plt.yticks(range(len(feature_importances_gb)), feature_importances_gb.index)
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()  

# %%

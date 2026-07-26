# python imports
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


#label for dataset columns -> unit id, flight cycle number, 3 engine setting and 21 sensor readings
columns = ["unitnum", "timecycle", "setting1", "setting2", "setting3"] \
        + [f"s_{i}" for i in range(1, 22)]

#read the text file into a DataFrame (table)
df = pd.read_csv("nasadata/train_FD001.txt", sep=r"\s+",
                 header=None, names=columns)

#sort rows by engine ID then by flight cycle
df = df.sort_values(["unitnum", "timecycle"]).reset_index(drop=True)


#for each engine find its last recorded cycle (= total lifespan)
max_lifetime = df.groupby("unitnum")["timecycle"].max().rename("max_lifetime")

#merge it back so every row knows its engines total lifespan
df = df.merge(max_lifetime, on="unitnum")

#RUL = total lifespan - current cycle
df["RUL"] = df["max_lifetime"] - df["timecycle"]

#cap RUL at 125 so that model doesnt get confused
df["RUL"] = df["RUL"].clip(upper=125)


#drop columns that never change (standard deviation = 0).
constant_cols = [col for col in df.columns if df[col].std() == 0]
print(f"dropping constant columns: {constant_cols}")
df.drop(columns=constant_cols, inplace=True)

#drop the helper column (dont need it anymore because we have RUL)
df.drop(columns=["max_lifetime"], inplace=True)


#now just checking the last 10 flight cycles
sensor_cols = [c for c in df.columns if c.startswith("s_")]
window = 10

#build all new features in a dict
new_features = {}

for col in sensor_cols:
    #         groupby engine so we calculate features per engine
    group = df.groupby("unitnum")[col]

    # rolling mean = average of the last 10 cycles
    new_features[f"{col}_mean"] = group.transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # rolling std = standard deviation of last 10
    new_features[f"{col}_std"] = group.transform(
        lambda x: x.rolling(window, min_periods=1).std().fillna(0)
    )

    # slope = steepness of trend of the sensor in last 10 cycles
    def get_slope(x):
        if len(x) < 2:
            return 0

        return np.polyfit(np.arange(len(x)), x, 1)[0]

    new_features[f"{col}_slope"] = group.transform(
        lambda x: x.rolling(window, min_periods=2).apply(get_slope, raw=True)
    )

    # diff = between min and max

    new_features[f"{col}_diff"] = group.diff().fillna(0)

    # min and max within the window = the range of values
    new_features[f"{col}_min"] = group.transform(
        lambda x: x.rolling(window, min_periods=1).min()
    )
    new_features[f"{col}_max"] = group.transform(
        lambda x: x.rolling(window, min_periods=1).max()
    )

# add all new columns at once instead of one by one
df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

# cycle_ratio = what percentage of its life has this engine completed
max_lifetime = df.groupby("unitnum")["timecycle"].max().rename("max_lifetime_temp")
df = df.merge(max_lifetime, on="unitnum")
df["cycle_ratio"] = df["timecycle"] / df["max_lifetime_temp"]
df.drop(columns=["max_lifetime_temp"], inplace=True)

# drop the original raw sensor columns because we have the features we need
df.drop(columns=sensor_cols, inplace=True)


# separate features (X) from the target (y = RUL)
feature_cols = [c for c in df.columns
                if c not in ["unitnum", "timecycle", "RUL"]]

print(f"\ntotal features: {len(feature_cols)}")
print(f"total samples:  {len(df)}")
X = df[feature_cols]
y = df["RUL"]

# 80% trainingand 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# switched this part from Random Forest to XGBoost because
# Random Forest builds all trees independently then averages but XGBoost builds trees one at a time where each new tree fixes the mistakes the previous trees did

#using 500 trees with 5percent correction on 80% of training data and same percent on features
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
    # regression (predicting a number) and minimize squared error

)
model.fit(X_train, y_train)
# predict RUL for the test set
y_pred = model.predict(X_test)
# RMSE = root mean squared error calculation looking for below 10 for a success
rmse = root_mean_squared_error(y_test, y_pred)
print(f"\nxgboost RMSE: {rmse:.2f} flight cycles")

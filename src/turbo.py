#python imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# labeling data and the column names with its unit ID, time in cycles, and the 3 engine settings, and 21 sensor reads
columns = ["unitnum", "timecycle", "setting1", "setting2", "setting3"] \
        + [f"s_{i}" for i in range(1, 22)]

# read the text file into a df (DataFrame).
df = pd.read_csv("nasadata/train_FD001.txt", sep=r"\s+",
                 header=None, names=columns)

# find each engines last cycle or max lifetime
max_lifetime = df.groupby("unitnum")["timecycle"].max().rename("max_lifetime")

# bring max_lifetime back into df so every row knows its engines total lifespan
df = df.merge(max_lifetime, on="unitnum")

# RUL = total lifespan minus the current cycle number
# meaning a brand-new engine (cycle 1) gets a high  and its the opposite for a dying one
df["RUL"] = df["max_lifetime"] - df["timecycle"]

# cap RUL at 125 so model doesnt get confuseds
df["RUL"] = df["RUL"].clip(upper=125)

# drop the columns that never change -> variance == 0
constant_cols = [col for col in df.columns
                 if df[col].std() == 0]
print(f"Dropping constant columns: {constant_cols}")
df.drop(columns=constant_cols, inplace=True)

# now we can drop the helper column we created because it was only needed to calculate RUL
df.drop(columns=["max_lifetime"], inplace=True)

#we have to see how each sensor has behaved in the last 5 cycles

sensor_cols = [c for c in df.columns if c.startswith("s_")]
window = 5

for col in sensor_cols:
    # rolling mean = the smoothed sensor value
    df[f"{col}_mean_{window}"] = (
        df.groupby("unitnum")[col]
          .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    # rolling std = how volatile the sensor is
    df[f"{col}_std_{window}"] = (
        df.groupby("unitnum")[col]
          .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )


#original settings + the new rolling features
feature_cols = [c for c in df.columns
                if c not in ["unitnum", "timecycle", "RUL"]]

print(f"\ntotal features: {len(feature_cols)}")
print(f"Total samples:  {len(df)}")


# seperate the features (X) from the target (y = RUL)
X = df[feature_cols]
y = df["RUL"]

# 80% for training adn 20% for testing it.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RandomForestRegressor is going to build 100 decision trees on random subsets of the data and average predictions
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# now we make the predictions on the testing data.
y_pred = model.predict(X_test)

# find the Root Mean Squared Error (RMSE). looking for a lower number here (fingers crossed)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"\nModel RMSE: {rmse:.2f} flight cycles")

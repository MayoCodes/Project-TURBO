# Project-TURBO

A machine learning model that predicts the remaining useful life (RUL) of jet engines using sensor data. Built with XGBoost and trained on NASAs turbofan degradation dataset.

## How It Works

T.U.R.B.O. works by taking in raw sensor data from jet engines in NASA's C-MAPSS
dataset and is made with Python with libraries like pandas, numpy, scikit-learn, and a
XGBoost Regressor model to predict how many flight cycles an engine has before it
breaks down. First, the code cleans up the data by removing useless, motionless
sensors and calculates trends like a ten-cycle average and slope to catch early signs
of engine wear. Then, an XGBoost machine learning model learns these wear patterns
to estimate an engine's Remaining Useful Life, aka it’s RUL. To test the model, I built a
supplementary real-world application system that utilizes the model and compares
the model's predictions with actual results to calculate model accuracy as well as give
out alerts relating to individual engine statuses to allow teams the time to allocate
their resources to fixing that engine before danger strikes

## files

- `src/turbo.py` — loads the NASA data, engineers features, trains the XGBoost model, and prints the RMSE score
- `src/simulateFlight.py` — uses the trained model to simulate a flight operations dashboard, showing engine health statuses across multiple engines
- `nasadata/train_FD001.txt` — NASA turbofan engine degradation dataset (FD001)

## requirements

```
pandas
numpy
xgboost
scikit-learn
```

install them with:

```
pip install -r requirements.txt
```

## how to run

from the project root:

```
python src/turbo.py
```

to see the simulation:

```
python src/simulateFlight.py
```

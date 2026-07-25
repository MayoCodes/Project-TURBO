#python imports
import pandas as pd
import numpy as np

#data headers
columns = ["unitnum", "timecycle", "setting1", "setting2", "setting3"] + [f{"s_{i}"} for i in range(1,22)]

#load dataset
df = pd.read_csv("nasadata/trainFD001.txt", sep="r\s+", header=None, names=columns)

import pandas as pd
import numpy as np
raw_feat = pd.read_csv("Data/Features.csv", index_col = 0)
raw_feat = raw_feat.drop('date', 1)
all_pm10 = []
all_so2 = []
all_no2 = []
all_o3 = []
asa = 0
holder = {0:all_pm10,1:all_so2,2:all_no2,3:all_o3}

for data in raw_feat:
    holder[asa]+=data
    asa+=1
    asa = asa%4




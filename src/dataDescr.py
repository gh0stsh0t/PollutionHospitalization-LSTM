import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def minMaxer(items):
    holder = []
    for item in items:
        holder.append((item.max(), item.min(), item.mean()))
    return holder

def grapher(partial):
    plt.style.use('seaborn-darkgrid')
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("µg/Ncm")
    ax1.set_xlabel("day")
    for station in partial:
        ax1.plot(station)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Hospitalizations")
    ax2.fill_between(range(0,103), targets, alpha=0.2)
    ax2.tick_params(axis='y')
    fig.tight_layout()
    plt.show()

raw_feat = pd.read_csv("Data/ProcessedData.csv", index_col = 0)
print(raw_feat)
raw_feat[list(raw_feat)] = raw_feat[list(raw_feat)].replace({-1:np.nan})
dateList = raw_feat['date']
raw_feat = raw_feat.drop('date', 1)
targets = raw_feat['target']
raw_feat = raw_feat.drop('target', 1)
print(raw_feat)
all_pm10 = []
all_so2 = []
all_no2 = []
all_o3 = []
asa = 0
pollus = [all_pm10,all_so2,all_no2,all_o3]

for name, vals in raw_feat.iteritems():
    pollus[asa].append(vals)
    asa += 1
    asa = asa%4

print(len(pollus[0]))
minMaxes = []
for pollu in pollus:
    minMaxes.append(minMaxer(pollu))
print(minMaxes)
print("")
stations = [2, 7, 11, 14]
pollutant = ['pm10', 'so2', 'no2', 'o3']

for count, val in enumerate(minMaxes):
    for counter, stat in enumerate(val):
        print("Station {:>2} {:>4} - Max: {:6} Min: {:>5} Mean: {:.3f}".format(stations[counter], pollutant[count], stat[0], stat[1], stat[2]))
for pollu in pollus:
    grapher(pollu)

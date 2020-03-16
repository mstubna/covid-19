from operator import itemgetter
import pathlib

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# constants
dirname = pathlib.Path(__file__).resolve().parent
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
colors = {
  'very_light_gray': '#ececec',
  'lighter_gray': '#dcdbdb',
  'light_gray': '#b6b6b6',
  'medium_gray': '#929292',
  'dark_gray': '#858585',
  'darker_gray': '#696969',
  'very_dark_gray': '#414141',
  'orange': '#ff6f00',
  'blue': '#1a237e',
  'very_light_blue': '#bbdefb',
  'very_light_purple': '#f3e1fc',
}
start_date = np.datetime64('2020-01-22')
all_dates = [start_date + np.timedelta64(x, 'D') for x in range(0, 100)]

data = pd.read_csv(dirname / 'CSSEGISandData--COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

def convert_to_ts (data, country):
  df = pd.DataFrame(data[data['Country/Region'] == country].reset_index().sum(
    axis=0, numeric_only=True
  )[3:]).reset_index()
  df.columns = ['date', 'count']
  df['date'] = df['date'].astype('datetime64[ns]')
  return df

china = convert_to_ts(data, 'China')
it = convert_to_ts(data, 'Italy')
iran = convert_to_ts(data, 'Iran')
sk = convert_to_ts(data, 'Korea, South')
esp = convert_to_ts(data, 'Spain')
fr = convert_to_ts(data, 'France')
us = convert_to_ts(data, 'US')

## figure 1
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

ax.plot(
  china['date'],
  china['count'],
  label='China',
  color=colors['very_light_gray']
)

sk_start_ind = 33
ax.plot(
  sk['date'] - np.timedelta64(sk_start_ind, 'D'),
  sk['count'],
  label=f'S. Korea = China - {sk_start_ind} days',
  color=colors['light_gray']
)

it_start_ind = 37
ax.plot(
  it['date'] - np.timedelta64(it_start_ind, 'D'),
  it['count'],
  label=f'Italy = China - {it_start_ind} days',
  color=colors['medium_gray']
)

iran_start_ind = 38
ax.plot(
  iran['date'] - np.timedelta64(iran_start_ind, 'D'),
  iran['count'],
  label=f'Iran = China - {iran_start_ind} days',
  color=colors['very_light_blue']
)

us_start_ind = 47
ax.plot(
  us['date'] - np.timedelta64(us_start_ind, 'D'),
  us['count'], label=f'US = China - {us_start_ind} days',
  linewidth=2,
  color=colors['orange']
)

plt.xlim((np.datetime64('2020-01-01'),np.datetime64('2020-03-15')))
plt.ylim((0, 30000))
ax.set_xticklabels([])
ax.set_yticklabels(['0' if x == 0 else '{:.0f}k'.format(int(x) / 1000) for x in ax.get_yticks().tolist()])
plt.legend(loc='lower right')

plt.show()

#%% figure 2
def sigmoid (x, A, h, slope, C):
  return 1 / (1 + np.exp ((x - h) / slope)) *  A + C

def fit_to_sigmoid (df, all_dates):
  dates = (df['date'] - start_date) / np.timedelta64(1, 'D')
  p, _ = curve_fit(
    sigmoid,
    dates,
    df['count'],
    p0=[ 80000, 17, -5, -1000],
    bounds=(
      [-np.inf, -np.inf, -np.inf, -np.inf],
      [np.inf, np.inf, -0.01, np.inf]
    ),
    maxfev=5000,
  )
  return sigmoid((all_dates - start_date) / np.timedelta64(1, 'D'), *p), p

# Plots the data
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

fit_china, p = fit_to_sigmoid(china, all_dates)
print('china', *p)
ax.plot(
  china['date'],
  china['count'],
  label='China',
  color=colors['very_light_gray'],
  linewidth=3
)
ax.plot(
  all_dates,
  fit_china,
  color=colors['very_light_gray'],
  linestyle=':'
)

fit_sk, p = fit_to_sigmoid(sk, all_dates)
print('sk', *p)
ax.plot(
  sk['date'],
  sk['count'],
  label='S. Korea',
  color=colors['light_gray'],
  linewidth=3
)
ax.plot(
  all_dates,
  fit_sk,
  color=colors['light_gray'],
  linestyle=':'
)

fit_it, p = fit_to_sigmoid(it, all_dates)
print('it', *p)
ax.plot(
  it['date'],
  it['count'],
  label='Italy',
  color=colors['medium_gray'],
  linewidth=3
)
ax.plot(
  all_dates,
  fit_it,
  color=colors['medium_gray'],
  linestyle=':'
)

fit_esp, p = fit_to_sigmoid(esp, all_dates)
print('esp', *p)
ax.plot(
  esp['date'],
  esp['count'],
  label='Spain',
  color=colors['very_light_blue'],
  linewidth=3
)
ax.plot(
  all_dates,
  fit_esp,
  color=colors['very_light_blue'],
  linestyle=':'
)

fit_fr, p = fit_to_sigmoid(fr, all_dates)
print('fr', *p)
ax.plot(
  fr['date'],
  fr['count'],
  label='France',
  color=colors['very_light_purple'],
  linewidth=3
)
ax.plot(
  all_dates,
  fit_fr,
  color=colors['very_light_purple'],
  linestyle='--'
)

fit_us, p = fit_to_sigmoid(us, all_dates)
print('us', *p)
ax.plot(
  us['date'],
  us['count'],
  label='US',
  color=colors['orange'],
  linewidth=3
)
ax.plot(
  all_dates,
  fit_us,
  color=colors['orange'],
  linestyle=':'
)

plt.ylim((0, 200000))
ax.set_yticklabels(['0' if x == 0 else '{:.0f}k'.format(int(x) / 1000) for x in ax.get_yticks().tolist()])
plt.legend(loc='lower right')
plt.show()


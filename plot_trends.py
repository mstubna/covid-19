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
  'light_gray': '#b6b6b6',
  'medium_gray': '#929292',
  'very_dark_gray': '#414141',
  'orange': '#ff6f00',
  'light_blue': '#79c3ff',
  'light_purple': '#d88aff',
  'light_green': '#b4ec70',
  'light_yellow': '#fff27e',
}
start_date = np.datetime64('2020-01-22')
all_dates = [start_date + np.timedelta64(x, 'D') for x in range(0, 100)]
should_print = True

data = pd.read_csv(dirname / 'CSSEGISandData--COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

def convert_to_ts (data, country):
  df = pd.DataFrame(data[data['Country/Region'] == country].reset_index().sum(
    axis=0, numeric_only=True
  )[3:]).reset_index()
  df.columns = ['date', 'count']
  df['date'] = df['date'].astype('datetime64[ns]')
  return df

country_names = ['China', 'Italy', 'Iran', 'Korea, South', 'Spain', 'France', 'Germany', 'US']
countries = { key: convert_to_ts(data, key) for key in country_names}

#%% figure 1
dat = [
  ('China', 0, 'light_gray'),
  ('Korea, South', 33, 'medium_gray'),
  ('Italy', 37, 'very_dark_gray'),
  ('Iran', 38, 'light_blue'),
  ('Spain', 45, 'light_purple'),
  ('Germany', 45, 'light_green'),
  ('France', 45, 'light_yellow'),
  ('US', 47, 'orange')
]

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

for country_name, offset, color_key in dat:
  country = countries[country_name]
  ax.plot(
    country['date'] - np.timedelta64(offset, 'D'),
    country['count'],
    label=f'{country_name} = China - {offset} days',
    color=colors[color_key]
  )

plt.xlim((np.datetime64('2020-01-01'),np.datetime64('2020-03-01')))
plt.ylim((0, 30000))
ax.set_xticklabels([])
ax.set_yticklabels(['0' if x == 0 else '{:.0f}k'.format(int(x) / 1000) for x in ax.get_yticks().tolist()])
plt.legend(loc='lower right')

if should_print:
  plt.savefig(dirname / f'figures/growth_rate.png', bbox_inches='tight', dpi=300, format='png')

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

for country_name, offset, color_key in dat:
  country = countries[country_name]
  fit, p = fit_to_sigmoid(country, all_dates)
  print(country_name, *p)

  ax.plot(
    country['date'],
    country['count'],
    label=country_name,
    color=colors[color_key],
    linewidth=3
  )
  ax.plot(
    all_dates,
    fit,
    color=colors[color_key],
    linestyle=':'
  )

plt.ylim((0, 100000))
ax.set_yticklabels(['0' if x == 0 else '{:.0f}k'.format(int(x) / 1000) for x in ax.get_yticks().tolist()])
plt.legend(loc='lower right')

if should_print:
  plt.savefig(dirname / f'figures/growth_estimations.png', bbox_inches='tight', dpi=300, format='png')

plt.show()


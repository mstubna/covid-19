from operator import itemgetter
import pathlib

import numpy as np
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
}

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
us = convert_to_ts(data, 'US')

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
  color=colors['dark_gray']
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

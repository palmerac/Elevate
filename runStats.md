# Running Workout Statistics and Graphs
2023-08-30

- [Graphs](#graphs)
  - [Running Distance Frequency](#running-distance-frequency)
  - [Average Heart Rate Frequency](#average-heart-rate-frequency)
  - [Average Heart Rate versus Average Speed
    (km/h)](#average-heart-rate-versus-average-speed-kmh)
  - [Heart Rate / Speed over Time](#heart-rate-speed-over-time)
  - [Distance vs Average Heart Rate](#distance-vs-average-heart-rate)
  - [Duration vs Average Heart Rate](#duration-vs-average-heart-rate)
- [Final Running Stats](#final-running-stats)

<details open>
<summary>Code</summary>

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
from datetime import datetime
import warnings

plt.style.use('ggplot')
warnings.filterwarnings('ignore')

start = time.perf_counter()
```

</details>
<details open>
<summary>Code</summary>

``` python
df = pd.read_csv('HeartWatch-Workouts-20230718-to-20230822.csv')
```

</details>
<details open>
<summary>Code</summary>

``` python
df = df.drop(['Date', 'from', 'to', 'rpe', 'Load', 'bpm-lo', 'bpm-90%+-%', '90%+-mins',
              'bpm-80-90%-%', '80-90%-mins','bpm-70-80%-%', '70-80%-mins','bpm-60-70%-%',
              '60-70%-mins','bpm-50-60%-%', '50-60%-mins'], axis=1)

# Drop Run w/ Bear
df = df.drop(19)
```

</details>
<details open>
<summary>Code</summary>

``` python
# Fix Datetime Columns
df['ISO'] = pd.to_datetime(df['ISO'])
df['Duration'] = pd.to_timedelta(df['Duration'])
df['/km'] = pd.to_timedelta(df['/km'])

df.set_index('ISO', inplace=True) 

wklySUM = pd.DataFrame(df[df['Type'] == 'Running'].groupby(pd.Grouper(freq='W-SUN')).agg('sum'))
wklyAVG = pd.DataFrame(df[df['Type'] == 'Running'].groupby(pd.Grouper(freq='W-SUN')).agg('mean'))

# print(wklySUM.head())
# print(wklyAVG.head())
```

</details>
<details open>
<summary>Code</summary>

``` python
dfRun = df[df['Type'] == 'Running']
dfBike = df[df['Type'] == 'Cycling']
dfOther = df[~df['Type'].isin(['Running', 'Cycling'])]

print(dfRun.info())
# print(dfBike.info())
# print(dfOther.head())
```

</details>

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 18 entries, 2023-07-19 11:53:49-04:00 to 2023-08-22 10:02:41-04:00
    Data columns (total 10 columns):
     #   Column    Non-Null Count  Dtype          
    ---  ------    --------------  -----          
     0   Duration  18 non-null     timedelta64[ns]
     1   Type      18 non-null     object         
     2   bpm-Avg.  18 non-null     float64        
     3   bpm-%     18 non-null     float64        
     4   bpm-hi    18 non-null     float64        
     5   Cals      18 non-null     float64        
     6   Cals/h    18 non-null     float64        
     7   km        18 non-null     float64        
     8   km/h      18 non-null     float64        
     9   /km       18 non-null     timedelta64[ns]
    dtypes: float64(7), object(1), timedelta64[ns](2)
    memory usage: 1.5+ KB
    None

<details open>
<summary>Code</summary>

``` python
# Pace Calculation
# Calculate the total seconds of Duration column
total_seconds = dfRun['Duration'].dt.total_seconds()

# Average Pace from M/S
mps = dfRun['km'].sum()*1000 / total_seconds.sum()
kph = mps * 3.6
mpk = 60 / kph

integer_part = int(mpk)
decimal_part = mpk - integer_part

# Convert decimal part to minutes by dividing by 60
decimal_minutes = round(decimal_part * 60,0)
```

</details>
<details open>
<summary>Code</summary>

``` python
# Weighted HR
dfRunWght = dfRun

# Convert the time delta to decimal hours and create a new column
dfRunWght["Duration"] = dfRunWght["Duration"].apply(lambda x: x.total_seconds() / 3600)
dfRunWght['Weighted HR'] = dfRunWght['Duration'] * dfRunWght['bpm-Avg.']

# HR/Speed Decimal
dfRun['HR/Speed'] = (dfRun['bpm-Avg.'] / dfRun['km/h']) 
```

</details>
<details open>
<summary>Code</summary>

``` python
# df['ISO'] = df['ISO'].dt.date
```

</details>

## Graphs

### Running Distance Frequency

<details open>
<summary>Code</summary>

``` python
rdf_bins = int(dfRun['km'].max() - dfRun['km'].min())

plt.hist(dfRun['km'], edgecolor='black', bins=rdf_bins)
plt.title('Run Distance Frequency')
plt.xlabel('Distance')
plt.ylabel('Frequency')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Graphs/Distance Frequency.png', dpi=300)
plt.show()
```

</details>

![](runStats_files/figure-commonmark/cell-10-output-1.png)

### Average Heart Rate Frequency

<details open>
<summary>Code</summary>

``` python
hrf_bins = int((dfRun['bpm-Avg.'].max() - dfRun['bpm-Avg.'].min())/2.5)

plt.hist(dfRun['bpm-Avg.'], edgecolor='black', bins=hrf_bins)
plt.title('Average HR Frequency')
plt.xlabel('Average BPM')
plt.ylabel('Frequency')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Graphs/HR Frequency.png', dpi=300)
plt.show()
```

</details>

![](runStats_files/figure-commonmark/cell-11-output-1.png)

### Average Heart Rate versus Average Speed (km/h)

<details open>
<summary>Code</summary>

``` python
# Line of Best Fit
# Fit a linear regression line to the data
degree = 1
coefficients = np.polyfit(dfRun['bpm-Avg.'], dfRun['km/h'], degree)
slope = coefficients[0]
intercept = coefficients[1]
# Calculate the predicted y-values using the line equation
predicted_y = slope * dfRun['bpm-Avg.'] + intercept
equation = f'y = {slope:.2f}x + {intercept:.2f}'

# Plot
plt.scatter('bpm-Avg.', 'km/h', data=dfRun)
plt.plot(dfRun['bpm-Avg.'], predicted_y, color='blue', label='Line of Best Fit')
plt.xlabel('Average BPM')
plt.ylabel('Average Speed (km/h)')
plt.title('Average HR versus Speed')
plt.text(0.01, 0.81, equation, fontsize=11, transform=plt.gca().transAxes)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Graphs/SpeedvsHR.png', dpi=300)
plt.show()
```

</details>

![](runStats_files/figure-commonmark/cell-12-output-1.png)

### Heart Rate / Speed over Time

Lower is better

<details open>
<summary>Code</summary>

``` python
plt.scatter(dfRun.index, y= dfRun['HR/Speed'])
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Avg HR/Speed')
plt.title('Avg HR/Speed over Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('Graphs/HR-Speed over Time.png', dpi=300)
plt.show()
```

</details>

![](runStats_files/figure-commonmark/cell-13-output-1.png)

### Distance vs Average Heart Rate

<details open>
<summary>Code</summary>

``` python
plt.scatter('km', 'bpm-Avg.', data=dfRun)
plt.xlabel('Distance')
plt.ylabel('Average Heart Rate')
plt.title('Distance vs Average HR')
plt.grid(True)
plt.tight_layout()
plt.savefig('Graphs/Distance vs Avg HR.png', dpi=300)
```

</details>

![](runStats_files/figure-commonmark/cell-14-output-1.png)

### Duration vs Average Heart Rate

<details open>
<summary>Code</summary>

``` python
plt.scatter('Duration', 'bpm-Avg.', data=dfRun)
plt.xlabel('Duration (Hours)')
plt.ylabel('Average Heart Rate')
plt.title('Duration vs Average HR')
plt.grid(True)
plt.tight_layout()
plt.savefig('Graphs/Duration vs Avg HR.png', dpi=300)
```

</details>

![](runStats_files/figure-commonmark/cell-15-output-1.png)

<details open>
<summary>Code</summary>

``` python
wd_ct = len(wklySUM)/1.5

plt.bar(wklySUM.index, wklySUM['km'], label='km', width=wd_ct)

for idx, value in enumerate(wklySUM['km']):
    plt.text(wklySUM.index[idx], 0, f'{value:.2f}', ha='center', va='bottom',
             fontsize=10, color='black')
    
plt.xlabel('Week')
plt.ylabel('KM')
plt.title('Weekly Distance')
plt.xticks(wklySUM.index, wklySUM.index.strftime('%Y-%m-%d'), rotation=45)
plt.tight_layout()
plt.savefig('Graphs/Weekly Distance.png', dpi=300)
plt.show()
```

</details>

![](runStats_files/figure-commonmark/cell-16-output-1.png)

<details open>
<summary>Code</summary>

``` python
# Average Duration, Distance, Average HR, Average Max HR Average Calories
avg_dist = round(dfRun['km'].mean(),2)
avg_hr = round(dfRun['bpm-Avg.'].mean(),2)
avg_wght_hr = round(dfRunWght['Weighted HR'].sum() / dfRunWght['Duration'].sum(),2)
avg_maxhr = round(dfRun['bpm-hi'].mean(),2)
avg_cals = round(dfRun['Cals'].mean(),2)
avg_dur = dfRun['Duration'].mean()

# Count Runs
num_runs = dfRun['km'].count()
ovr_5k = dfRun[dfRun['km'] >=5].count()['km']
povr_5k = round(ovr_5k / num_runs *100,2)
ovr_10k = dfRun[dfRun['km'] >=10].count()['km']
povr_10k = round(ovr_10k / num_runs *100,2)

# Maximums
max_dur = dfRun['Duration'].max()
max_dist = dfRun['km'].max()
max_avghr = dfRun['bpm-Avg.'].max()
max_maxhr = dfRun['bpm-hi'].max()
max_cals = dfRun['Cals'].max()

# Totals
tot_dist = round(dfRun['km'].sum(),2)
tot_dur = dfRun['Duration'].sum()
tot_cals = round(dfRun['Cals'].sum(),2)

# Medians
med_dist = round(dfRun['km'].median(),2)
med_avg_hr = round(dfRun['bpm-Avg.'].median(),2)
med_max_hr = round(dfRun['bpm-hi'].median(),2)
med_cals = round(dfRun['Cals'].median(),2)

# Durations to Time Format
avg_dur_h = int(avg_dur)
max_dur_h = int(max_dur)
tot_dur_h = int(tot_dur)

avg_dur_m_dec = (avg_dur - avg_dur_h)*60
max_dur_m_dec = (max_dur - max_dur_h)*60
tot_dur_m_dec = (tot_dur - tot_dur_h)*60

avg_dur_m = int(avg_dur_m_dec)
max_dur_m = int(max_dur_m_dec)
tot_dur_m = int(tot_dur_m_dec)

avg_dur_s = int((avg_dur_m_dec - avg_dur_m)*60)
max_dur_s = int((max_dur_m_dec - max_dur_m)*60)
tot_dur_s = int((tot_dur_m_dec - tot_dur_m)*60)

avg_dur_f = str(avg_dur_h) + ':' + str(avg_dur_m) + ':' + str(avg_dur_s)
max_dur_f = str(max_dur_h) + ':' + str(max_dur_m) + ':' + str(max_dur_s)
tot_dur_f = str(tot_dur_h) + ':' + str(tot_dur_m) + ':' + str(tot_dur_s)
```

</details>

## Final Running Stats

<details open>
<summary>Code</summary>

``` python
print(f'Runs: {num_runs}')
print(f'Runs over 5k(%): {ovr_5k} ({povr_5k}%)')
print(f'Runs over 10k(%): {ovr_10k} ({povr_10k}%)')

print('----------------------------')
print(f'Average Duration: {avg_dur_f}')
print(f'Average Distance: {avg_dist}')
print(f"Average Pace: {integer_part}:{decimal_minutes}")
print(f'Average Weighted HR: {avg_wght_hr}')
print(f'Average HR: {avg_hr}')
print(f'Average Max HR: {avg_maxhr}')
print(f'Average Calories: {avg_cals}')

print('----------------------------')
print(f'Max Duration: {max_dur_f}')
print(f'Max Distance: {max_dist}')
print(f'Max Average HR: {max_avghr}')
print(f'Max Max HR: {max_maxhr}')
print(f'Max Calories: {max_cals}')

print('----------------------------')
print(f'Median Distance: {med_dist}')
print(f'Median Avg HR: {med_avg_hr}')
print(f'Median Max HR: {med_max_hr}')
print(f'Median Calories: {med_cals}')

print('----------------------------')
print(f'Total Duration: {tot_dur_f}')
print(f'Total Distance: {tot_dist}')
print(f'Total Calories Burn: {tot_cals}')

print('----------------------------')
print(f'Runtime: {round(time.perf_counter() - start,2)}s')
```

</details>

    Runs: 18
    Runs over 5k(%): 14 (77.78%)
    Runs over 10k(%): 1 (5.56%)
    ----------------------------
    Average Duration: 0:46:0
    Average Distance: 6.32
    Average Pace: 7:17.0
    Average Weighted HR: 151.45
    Average HR: 151.61
    Average Max HR: 170.11
    Average Calories: 517.15
    ----------------------------
    Max Duration: 1:21:26
    Max Distance: 12.61
    Max Average HR: 160.2
    Max Max HR: 191.0
    Max Calories: 1008.3
    ----------------------------
    Median Distance: 5.96
    Median Avg HR: 151.35
    Median Max HR: 167.0
    Median Calories: 489.65
    ----------------------------
    Total Duration: 13:48:7
    Total Distance: 113.82
    Total Calories Burn: 9308.7
    ----------------------------
    Runtime: 1.85s

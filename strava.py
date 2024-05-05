import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import streamlit as st
import numpy as np
from datetime import timedelta
import seaborn as sns

plt.style.use("ggplot")
st.set_page_config(layout="wide")

st.sidebar.markdown("## Upload your own data")
st.sidebar.markdown('Export standard Workouts via Heartwatch app')
uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("HeartWatch-Workouts-20230717-to-20231106.csv")

df = df.drop(
    [
        "Date",
        "from",
        "to",
        "rpe",
        "bpm-%",
        "bpm-90%+-%",
        "90%+-mins",
        "bpm-80-90%-%",
        "80-90%-mins",
        "bpm-70-80%-%",
        "70-80%-mins",
        "bpm-60-70%-%",
        "60-70%-mins",
        "bpm-50-60%-%",
        "50-60%-mins",
    ],
    axis=1,
)


df["ISO"] = pd.to_datetime(df["ISO"])
df["Duration"] = df["Duration"].astype(str)


def convert_to_seconds(time_string):
    hours, minutes, seconds = map(int, time_string.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def convert_to_time_string(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if hours == 0:
        return f"{minutes} min, {seconds} sec"
    elif minutes == 0:
        return f"{hours} hours, {seconds} sec"
    elif seconds == 0:
        return f"{hours} hours, {minutes} min"
    else:
        return f"{hours} hours, {minutes} min, {seconds} sec"


def avghr(df):
    beats = df.totalBeats.sum()
    dur = df.mDuration.sum()
    bpm = round(beats / dur, 2)
    return bpm

df["sDuration"] = df["Duration"].apply(convert_to_seconds)
df["mDuration"] = df["sDuration"] / 60
df["hDuration"] = df["mDuration"] / 60

df["totalBeats"] = round(
    df["mDuration"] * df["bpm-Avg."],
)

run = df[df["Type"] == "Running"]
bike = df[df["Type"] == "Cycling"]
walk = df[df["Type"] == "Walking"]
golf = df[df["Type"] == "Golf"]

for g in [run, bike, walk, golf]:
    g["cumkm"] = g["km"].cumsum()
    g["cumdur"] = g['hDuration'].cumsum()

# Streamlit main page
st.title("Workout Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.write(f"Total Activies: {len(df)}")
    st.write(f"Total Duration: {convert_to_time_string(df.sDuration.sum())}")
    st.write(f"Total Distance: {df.km.sum()} km")
    st.write(f"Total Heart Beats: {df.totalBeats.sum():,.0f}")
    st.write(f"Total Calories Burned: {df.Cals.sum():,.0f}")
with col2:
    st.markdown('\n')
    st.write(f"Average Duration: {convert_to_time_string(df.sDuration.sum()/len(df))}")
    st.write(f"Average Distance: {round(df.km.mean(),2)} km")
    st.write(f"Average BPM: {avghr(df)}")
    st.write(f"Average Calories Burned: {round(df.Cals.sum() / len(df),2)}")
    st.write(f"Average Calories/Hour: {round(df.Cals.sum() / df.hDuration.sum(),2)}")

# Activity pages
def display_summary_and_raw_data(data, title):
    st.subheader(title)
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Distributions", "Charts", "Raw Data"])
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Total {title}: {len(data)}")
            st.write(f"Total Duration: {convert_to_time_string(data.sDuration.sum())}")
            st.write(f"Total Distance: {data.km.sum()} km")
            st.write(f"Total Heart Beats: {data.totalBeats.sum():,.0f}")  
            st.write(f"Total Calories Burned: {data.Cals.sum():,.0f}")  

        with col2:
            st.markdown("#")
            st.write(
                f"Average Duration: {convert_to_time_string(data.sDuration.sum()/len(data))}"
            )
            st.write(f"Average Distance: {round(data.km.mean(),2)} km")
            st.write(f"Average BPM: {avghr(data)}")
            st.write(f"Average Calories Burned: {round(data.Cals.mean(),2)}")
        with col3:
            st.markdown("#")
            st.write(f"Median Duration {convert_to_time_string(data.sDuration.median())}")
            st.write(f"Median Distance: {round(data.km.median(),2)} km")
            st.write(f"Median BPM: {data['bpm-Avg.'].median()}")
            st.write(f"Median Calories Burned: {data.Cals.median():.0f}")
    with tab2:
        with st.expander("Boxplot"):
            columns = ['km', 'mDuration', 'bpm-Avg.', 'bpm-hi', 'km/h', 'Cals/h']
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
            axes = axes.flatten()
            for i, column in enumerate(columns):
                sns.boxplot(x=data[column], ax=axes[i])
                axes[i].set_title(column)
            plt.suptitle('Boxplots', fontsize=20)
            plt.tight_layout()
            st.pyplot(fig)
        
        def hist(col):
            fig, ax = plt.subplots()
            ax.hist(data[col], bins=10, edgecolor='black')
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.yaxis.set_major_locator(MultipleLocator(1))
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Avg BPM"):
            hist('bpm-Avg.')

        with st.expander("Max BPM"):
            hist('bpm-hi')

        with st.expander("Distance"):
            hist('km')
        
        with st.expander("Duration (min)"):
            hist('mDuration')

        with st.expander("Speed (km/h)"):
            hist('km/h')

    with tab3:
        with st.expander('Cumulative Distance'):
            fig, ax = plt.subplots()
            ax.plot(data['ISO'], data['cumkm'])
            plt.title('Cumulative Distance')
            plt.ylabel('Distance (km)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander('Cumulative Duration'):
            fig, ax = plt.subplots()
            ax.plot(data['ISO'], data['cumdur'])
            plt.title('Cumulative Duration')
            plt.ylabel('Time (h)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander('Distance over Time'):
            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(data['cumkm'], data['km'], label='Distance', color='red')
            m, b = np.polyfit(data['cumkm'], data['km'], 1)
            ax.plot(data['cumkm'], m*data['cumkm'] + b, color='blue', label='Distance LOBF')
            ax.plot(data['cumkm'], data['km'].rolling(window=5).mean(), label=f'5 {title} distance MA', color='green', linestyle='--')
            ax.plot(data['cumkm'], data['km'].rolling(window=10).mean(), label=f'10 {title} speed MA', color='purple', linestyle='--')
            
            plt.title('Distance over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.xlabel('Cumulative KM')
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Speed over Time"):
            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(data['cumkm'], data['km/h'], label='Speed(km/h)', color='red')
            m, b = np.polyfit(data['cumkm'], data['km/h'], 1)
            ax.plot(data['cumkm'], m*data['cumkm'] + b, color='blue', label='Speed LOBF')
            ax.plot(data['cumkm'], data['km/h'].rolling(window=5).mean(), label=f'5 {title} speed MA', color='green', linestyle='--')
            ax.plot(data['cumkm'], data['km/h'].rolling(window=10).mean(), label=f'10 {title} speed MA', color='purple', linestyle='--')
            plt.title('Speed over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.xlabel('Cumulative KM')
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Average BPM over Time"):
            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(data['cumkm'], data['bpm-Avg.'], label='HR (BPM)', color='red')
            m, b = np.polyfit(data['cumkm'], data['bpm-Avg.'], 1)
            ax.plot(data['cumkm'], m*data['cumkm'] + b, color='blue', label='BPM LOBF')
            ax.plot(data['cumkm'], data['bpm-Avg.'].rolling(window=5).mean(), label=f'5 {title} BPM MA', color='green', linestyle='--')
            ax.plot(data['cumkm'], data['bpm-Avg.'].rolling(window=10).mean(), label=f'10 {title} BPM MA', color='purple', linestyle='--')
            plt.title('Average BPM over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.xlabel('Cumulative KM')
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Duration over Time"):
            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(data['cumkm'], data['mDuration'], label='Duration (min)', color='red')
            m, b = np.polyfit(data['cumkm'], data['mDuration'], 1)
            ax.plot(data['cumkm'], m*data['cumkm'] + b, color='blue', label='Duration LOBF')
            ax.plot(data['cumkm'], data['mDuration'].rolling(window=5).mean(), label=f'5 {title} Duration MA', color='green', linestyle='--')
            ax.plot(data['cumkm'], data['mDuration'].rolling(window=10).mean(), label=f'10 {title} Duration MA', color='purple', linestyle='--')
            plt.title('Duration over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.xlabel('Cumulative KM')
            plt.tight_layout()
            st.pyplot(fig)
    with tab4:
        st.dataframe(data)

# Usage:
tabs = ["Run", "Bike", "Walk", "Golf"]  # Add more tabs as needed
data_dict = {
    "Run": run,
    "Bike": bike,
    "Walk": walk,
    "Golf": golf,
}  

tab_run, tab_bike, tab_walk, tab_golf = st.tabs(tabs)  # Create main tabs

with tab_run:
    display_summary_and_raw_data(data_dict["Run"], "Runs")

with tab_bike:
    display_summary_and_raw_data(data_dict["Bike"], "Rides")

with tab_walk:
    display_summary_and_raw_data(data_dict["Walk"], "Walks")

with tab_golf:
    display_summary_and_raw_data(data_dict["Golf"], "Golf")

st.markdown('\n')
st.markdown('---')
st.markdown("Made with ❤️ by [palmerac](https://github.com/palmerac)")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Read in CSV and set title
df = pd.read_csv("runStats.csv")
st.title("Run Stats Dashboard")


def avg(df, col, rnd):
    ans = round(df[col].mean(), rnd)
    return ans


def med(df, col, rnd):
    ans = round(df[col].median(), rnd)
    return ans


def tot(df, col, rnd=None):
    ans = round(df[col].sum(), rnd)
    return ans


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Weekly Distance",
        "Boxplots",
        "Distance Histogram",
        "Avg HR Histogram",
        "Cumulative Distance",
        "Raw File",
    ]
)
with tab1:
    st.image("Graphs/Weekly Distance.png")
    pass

with tab2:
    st.image("Graphs/Run Boxplots.png")
    pass
with tab3:
    st.image("Graphs/Distance Frequency.png")
    pass

with tab4:
    st.image("Graphs/HR Frequency.png")
    pass

with tab5:
    st.image("Graphs/Cumulative Distance.png")
    pass

with tab6:
    st.dataframe(df)
    pass
st.subheader("Running Statistics")

# Create four columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Averages")
    st.write("Average Distance(km):", avg(df, "km", 2))
    st.write("Average Calories Burned:", avg(df, "Cals", None))
    st.write("Average Duration(min):", avg(df, "mDuration", 2))
    st.write("Average Heart Rate:", avg(df, "bpm-Avg.", 1))
    st.write("Average Max Heart Rate:", avg(df, "bpm-hi", 1))
    st.write("Average Speed(km/h):", avg(df, "km/h", 2))
    pass

with col2:
    st.subheader("Medians")
    st.write("Median Distance(km):", med(df, "km", 2))
    st.write("Median Calories Burned:", med(df, "Cals", None))
    st.write("Median Duration(min):", med(df, "mDuration", 2))
    st.write("Median Heart Rate:", med(df, "bpm-Avg.", 1))
    st.write("Median Max Heart Rate:", med(df, "bpm-hi", 1))
    st.write("Median Speed(km/h):", med(df, "km/h", 2))
    pass

num_runs = df["km"].count()
avghr_u155 = df[df["bpm-Avg."] < 155].count()["bpm-Avg."]
pavghr_u155 = round(avghr_u155 / num_runs * 100, 1)
avghr_u150 = df[df["bpm-Avg."] < 150].count()["bpm-Avg."]
pavghr_u150 = round(avghr_u150 / num_runs * 100, 1)
ovr_5k = df[df["km"] >= 5].count()["km"]
povr_5k = round(ovr_5k / num_runs * 100, 1)
ovr_10k = df[df["km"] >= 10].count()["km"]
povr_10k = round(ovr_10k / num_runs * 100, 1)

with col3:
    st.subheader("Counts")
    st.write("Total Runs:", df["km"].count())
    st.write("Runs > 10k (%):", ovr_10k, " (", povr_10k, ")")
    st.write("Runs > 5k (%):", ovr_5k, " (", povr_5k, ")")
    st.write("Runs < 155bpm (%):", avghr_u155, " (", pavghr_u155, ")")
    st.write("Runs < 150bpm (%):", avghr_u150, " (", pavghr_u150, ")")

with col4:
    st.subheader("Totals")
    st.write("Total Distance(km):", tot(df, "km", 2))
    st.write("Total Calories Burned:", tot(df, "Cals", None))
    st.write("Total Duration(h):", tot(df, "Duration", 2))

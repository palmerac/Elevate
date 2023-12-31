import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Read in CSV and set title
df = pd.read_csv("bikeStats.csv")
st.title("Bike Stats Dashboard")


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
    st.image("Graphs/BWeekly Distance.png")
    pass

with tab2:
    st.image("Graphs/Bike Boxplots.png")
    pass

with tab3:
    st.image("Graphs/BDistance Frequency.png")
    pass

with tab4:
    st.image("Graphs/BHR Frequency.png")
    pass

with tab5:
    st.image("Graphs/BCumulative Distance.png")
    pass

with tab6:
    st.dataframe(df)
    pass

st.subheader("Statistics")

# Create two columns
col1, col2, col3 = st.columns(3)

# Put content in the first column
with col1:
    st.subheader("Averages")
    st.write("Average Distance(km):", avg(df, "km", 2))
    st.write("Average Calories Burned:", avg(df, "Cals", None))
    st.write("Average Duration(min):", avg(df, "mDuration", 2))
    st.write("Average Heart Rate:", avg(df, "bpm-Avg.", 1))
    st.write("Average Max Heart Rate:", avg(df, "bpm-hi", 1))
    st.write("Average Speed(km/h):", avg(df, "km/h", 2))
    pass

# Put content in the second column
with col2:
    st.subheader("Medians")
    st.write("Median Distance(km):", med(df, "km", 2))
    st.write("Median Calories Burned:", med(df, "Cals", None))
    st.write("Median Duration(min):", med(df, "mDuration", 2))
    st.write("Median Heart Rate:", med(df, "bpm-Avg.", 1))
    st.write("Median Max Heart Rate:", med(df, "bpm-hi", 1))
    st.write("Median Speed(km/h):", med(df, "km/h", 2))
    pass
# Put content in the third column
with col3:
    st.subheader("Totals")
    st.write("Total Distance(km):", tot(df, "km", 2))
    st.write("Total Calories Burned:", tot(df, "Cals", None))
    st.write("Total Duration(h):", tot(df, "Duration", 2))
    st.write("Total Bike Rides:", df["km"].count())

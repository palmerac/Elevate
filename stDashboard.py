import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in CSV and set title
df = pd.read_csv("runStats.csv")
st.title("Run Stats Dashboard")


def avg(col, rnd):
    ans = round(df[col].mean(), rnd)
    return ans


def med(col, rnd):
    ans = round(df[col].median(), rnd)
    return ans


def tot(col, rnd=None):
    ans = round(df[col].sum(), rnd)
    return ans


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Weekly Distance",
        "Boxplots",
        "Distance Histogram",
        "Avg HR Histogram",
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
    st.dataframe(df)
    pass

# Create two columns
col1, col2, col3 = st.columns(3)

# Put content in the first column
with col1:
    st.subheader("Averages")
    st.write("Average Distance:", avg("km", 2))
    st.write("Average Calories Burned:", avg("Cals", 1))
    st.write("Average Duration:", avg("Duration", 2))
    st.write("Average Heart Rate:", avg("bpm-Avg.", 2))
    st.write("Average Max Heart Rate:", avg("bpm-hi", 2))
    st.write("Average Speed:", avg("km/h", 2))
    pass

# Put content in the second column
with col2:
    st.subheader("Medians")
    st.write("Median Distance:", med("km", 2))
    st.write("Median Calories Burned:", med("Cals", 1))
    st.write("Median Duration:", med("Duration", 2))
    st.write("Median Heart Rate:", med("bpm-Avg.", 2))
    st.write("Median Max Heart Rate:", med("bpm-hi", 2))
    st.write("Median Speed:", med("km/h", 2))
    pass
# Put content in the third column
with col3:
    st.subheader("Totals")
    st.write("Total Distance:", tot("km", 2))
    st.write(
        "Total Calories Burned:",
        tot(
            "Cals",
        ),
    )
    st.write("Total Duration:", tot("Duration", 2))
    st.write("Total Runs:", df["km"].count())
    pass

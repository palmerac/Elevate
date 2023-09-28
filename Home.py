import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfR = pd.read_csv("runStats.csv")
dfB = pd.read_csv("bikeStats.csv")

st.set_page_config(layout="wide")

st.sidebar.markdown(
    """
---
Created with ❤️ by [Aaron Palmer](https://github.com/palmerac/).
"""
)


def tot(col, rnd=None):
    total = round(dfR[col].sum() + dfB[col].sum(), rnd)
    return total


tot_rhr = dfR["bpm-Avg."].mean() * (60 * dfR["Duration"].sum())
tot_bhr = dfB["bpm-Avg."].mean() * (60 * dfB["Duration"].sum())
tot_hr = round(tot_rhr + tot_bhr, None)

st.title("Strava Dashboard")
st.markdown("\n  ")
st.text("This is a dashboard for my Apple Watch activities.")
st.text("It has running and cycling sections with more graphs and statistics.")
st.markdown("\n  ")

st.subheader("Total Weekly Distance")
with st.subheader("Total Weekly Distance"):
    st.image("Graphs/Total Weekly.png")
    pass

st.write("Total Distance:", tot("km", 2))
st.write("Total Duration:", tot("Duration", 2))
st.write("Total Calories:", tot("Cals", None))
st.write("Total Heart Beats:", tot_hr)
st.markdown("\n")
st.text("Made by: Aaron Palmer")

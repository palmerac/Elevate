import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_extras.streaming_write import write

dfR = pd.read_csv("runStats.csv")
dfB = pd.read_csv("bikeStats.csv")


def tot(col, rnd=None):
    total = round(dfR[col].sum() + dfB[col].sum(), rnd)
    return total


st.title("Strava Dashboard")
st.markdown("\n  ")

message = "This is a dashboard for my Apple Watch activities.  \n  \
            It has running and cycling sections with deeper analysis."


def stream():
    for word in message.split():
        yield word + " "
        time.sleep(0.1)


write(stream)

st.markdown("#### This is a dashboard for my Apple Watch activities.")
st.markdown("#### It has running and cycling sections with deeper analysis.")
st.markdown("\n  ")
st.write("Total Distance:", tot("km", 2))
st.write("Total Duration:", tot("Duration", 2))
st.write(
    "Total Calories:",
    tot(
        "Cals",
    ),
)
st.markdown("\n")
st.markdown("Made by: Aaron Palmer")

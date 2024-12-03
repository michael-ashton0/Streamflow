import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import CSC314_2024_Fall_MichaelAshton_Streamflow as Streamflow
from datetime import datetime, timedelta, date
import pandas as pd

#Define initial values
DEFAULT_STATION_ID = '11527000'

DEFAULT_DATE = date.today()

def graph_data(df, date, t1, t2):
    '''Graphs the Maximum, Minimum, and Current lines; +- 0.5 Standard Deviation area;
    as well as a prediction for any future data'''

    #Nothing new here this is just an adapted version of the method in my other code
    max_year = df.columns[1]
    min_year = df.columns[2]

    for column in df.columns[1:-1]:
        total_discharge = df[column].sum()

        if total_discharge > df[max_year].sum():
            max_year = column
        if total_discharge < df[min_year].sum():
            min_year = column

    df['avg_discharge'] = df.iloc[:, 1:].mean(axis=1)
    std_discharge = df.iloc[:, 1:].std(axis=1)

    mean_discharge = df['avg_discharge']
    df['avg_plus_0.5std'] = mean_discharge + (0.5 * std_discharge)
    df['avg_minus_0.5std'] = mean_discharge - (0.5 * std_discharge)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.yaxis.grid()
    ax.plot(df['datetimeUTC'], df[f'{max_year}'], color='orangered', label=f'Max Discharge Year')
    ax.plot(df['datetimeUTC'], df[f'{min_year}'], color='blue', label=f'Min Discharge Year')

    ax.plot(df['datetimeUTC'], df['avg_discharge'], color='gray', alpha=0)
    ax.plot(df['datetimeUTC'], df['avg_plus_0.5std'], color='gray', alpha=0.8, label='Average ± 0.5 stdev')
    ax.plot(df['datetimeUTC'], df['avg_minus_0.5std'], color='gray', alpha=0.8)
    ax.fill_between(df['datetimeUTC'], df['avg_plus_0.5std'], df['avg_minus_0.5std'], color='gray', alpha=0.4)

    plt.plot(df['datetimeUTC'], df['discharge_10'], color='black', label=f'Original Year')

    ax.axvline(x=pd.to_datetime(date - timedelta(weeks=52)), color='black', linestyle=':')

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    ax.set_title('River Stream Flow\n' + f'{t1} : {t2}', fontsize=10, loc='center')

    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (CFS)')
    ax.legend()
    plt.tight_layout()

    return fig

st.title("River Discharge Data Analysis")

station_id = st.text_input("Enter NWIS Station ID:", DEFAULT_STATION_ID)

st.caption('This is for the Trinity River at Burnt Ranch, other notable sensors are:')
st.caption('12399500 (Columbia River, International Border); 09380000 (Colorado River, Lees Ferry, AZ);')
st.caption('01315500 (Hudson River, North Creek, NY); 08330000 (Rio Grande, Albequerque, NM)')

selected_date = st.date_input("Select a Start Date:", DEFAULT_DATE)

if st.button("Generate Graph"):
    try:
        data = Streamflow.Streamflow(station_id, selected_date.strftime('%Y-%m-%d'))

        curr = data.get_current_year_data()
        curr = data.add_regression_data(curr)

        df = data.merge_data()

        st.subheader("Stream Flow Analysis")
        fig = graph_data(df, selected_date, data.total_flow(df), data.current_change(df))
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
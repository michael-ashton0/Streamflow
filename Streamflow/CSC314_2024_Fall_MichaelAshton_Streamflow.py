import pandas as pd
import matplotlib
import hydrofunctions as hf
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from scipy.integrate import quad
from datetime import timedelta

#### data processing tricks
# rename usgs column to current year DONE
# manually change indices to make them all the same year, otherwise x-axis goes per-year DONE
# can distinguish one year from another by using the renamed column to keep it organized under the correct year IN PROGRESS
# df.set/reset_index() can be used to take the index of times and change them to read all one year IN PROGRESS
# the way to do this is to create another index column, making the current index column just another column, 
# then you can add just one of them and drop all the others in order to create the df that has all years available to show in a continuous graph
# NOT SURE WHAT I EVEN MEANT BY THIS

### math
# Riemann Sum (area under the curve) of the current year up to current date will be the CFS flow intended to go right under the title DONE
# 'derivative' will be the change DONE
# linear regression after the current date to give a projection DONE

class Streamflow:
    def __init__(self, id, date):
        self.id = id
        self.date = pd.to_datetime(date)
        
    def get_single_year_data(self, date):
        '''Fetch discharge data for a single year and print it with a 'datetimeUTC' and 'discharge' column.'''

        date = pd.to_datetime(date)
        start_date = str(date - timedelta(weeks=2))[:10]    #Cut off the time
        end_date = str(date + timedelta(weeks=1))[:10]

        nwis_data = hf.NWIS(self.id, 'iv', 
                            start_date=start_date, 
                            end_date=end_date)
        
        discharge_data = nwis_data.df('discharge')
        #In case the above line is not cutting it uncomment this code for more robust error checking - still unsure how necessary it is
        '''try:
            discharge_data = nwis_data.df('discharge')
            if discharge_data.empty:
                discharge_data = nwis_data.df.get('Discharge') #One specific sensor was running into issues with this
            if discharge_data.empty:
                raise KeyError("Neither 'discharge' nor 'Discharge' found in the data.")
        except AttributeError as e:
            raise AttributeError("Error accessing data: Ensure nwis_data.df() is returning a DataFrame.") from e
        except KeyError as e:
            raise KeyError(str(e)) from e'''
        
        #Apply a consistent format
        discharge_data.index = pd.to_datetime(discharge_data.index)

        #Renaming to discharge for quality of life
        discharge_data = discharge_data.rename(columns={discharge_data.columns[0]: 'discharge'})
        
        #Make datetime the first column
        discharge_data.reset_index(inplace=True)
        discharge_data.rename(columns={'index': 'year'}, inplace=True)
        discharge_data.rename(columns={'datetimeUTC' : 'year'})
        
        return discharge_data
    
    def get_current_year_data(self):
        '''Queries and retrieves data for the given date and previous two weeks'''

        end_date = str(self.date)[:10]
        start_date = str(self.date - timedelta(weeks=2))[:10]

        nwis_data = hf.NWIS(self.id, 'iv', 
                            start_date=start_date, 
                            end_date=end_date)
        
        discharge_data = nwis_data.df('discharge')
        
        discharge_data.index = pd.to_datetime(discharge_data.index)

        discharge_data = discharge_data.rename(columns={discharge_data.columns[0]: 'discharge'})
        
        discharge_data.reset_index(inplace=True)
        discharge_data.rename(columns={'index': 'year'}, inplace=True)
        discharge_data.rename(columns={'datetimeUTC' : 'year'})

        return discharge_data
    
    def add_regression_data(self, df):
        '''Calculates and appends data corresponding to the linear regression of the current year
        in order to predict possible future flow'''

        #Convert timestamps to ordinal for calculation
        df['Ordinal'] = df['datetimeUTC'].map(pd.Timestamp.toordinal)
        coeffs = np.polyfit(df['Ordinal'], df['discharge'], deg=1)

        #Add the future timestamps
        last_timestamp = df['datetimeUTC'].iloc[-1]
        future_timestamps = pd.date_range(
                                start=last_timestamp + pd.Timedelta(minutes=15), 
                                periods=96*7, #96 15min intervals per day
                                freq='15min', 
                                tz='UTC')

        future_ordinals = future_timestamps.map(pd.Timestamp.toordinal)
        future_values = np.polyval(coeffs, future_ordinals)

        future_data = pd.DataFrame({
                        'datetimeUTC': future_timestamps,
                        'discharge': future_values})
        
        df_combined = pd.concat([df, future_data], ignore_index=True)

        return df_combined
    
    def merge_data(self):
        '''Fetches the data for the past 10 years and combines the data into a single dataframe'''
        dfs = []
        anchor_year = int(str(self.date)[:4])

        for year in range(anchor_year - 1, anchor_year - 10, -1):
            query_date = f'{year}{str(self.date)[4:]}'
            df = self.get_single_year_data(query_date)
            dfs.append(df)
    
        current = self.get_current_year_data()
        dfs.append(self.add_regression_data(current))
        
        merged_df = dfs[0]

        for i, df in enumerate(dfs[1:], start=2):
            df = df.rename(columns={'discharge': f'discharge_{i}'})
            merged_df = pd.merge(merged_df, df[[f'discharge_{i}']],
                                left_index=True,
                                right_index=True)
        return merged_df
    
    def current_change(self, df):
        '''Returns the comparison of the two most recent updates from the sensor'''
        change = df['discharge'].iloc[-2] - df['discharge'].iloc[-1]
        if change > 0:
            return f'Currently rising at {change} CFS'
        elif change < 0:
            change = int(abs(change))
            return f'Currently dropping by {change} CFS'
        else:
            return f'No reported change in flow over past 15 minutes'
        
    def total_flow(self, df):
        '''Effectively a Riemann Sum, returns the total volume of water that has passed
        through the river in a given year'''
        total = 0
        for value in df['discharge']:
            total += value * 0.25
        return f'{total} CFS within the time period'
    
    def graph_stream_data(self, df):
        '''Graphs the Maximum, Minimum, and Current lines; +- 0.5 Standard Deviation area;
        as well as a prediction for any future data'''

        #Set some initial values
        max_year = df.columns[1]
        min_year = df.columns[2]

        for column in df.columns[1:-1]:
            total_discharge = df[column].sum()

            if total_discharge > df[max_year].sum():
                max_year = column
            if total_discharge < df[min_year].sum():
                min_year = column
        
        #Exclude dates and times from the average
        df['avg_discharge'] = df.iloc[:, 1:].mean(axis=1, skipna=True)

        std_discharge = df.iloc[:, 1:].std(axis=1, skipna=True)

        #Calculate the bounds for ±0.5 standard deviations
        mean_discharge = df['avg_discharge']
        df['avg_plus_0.5std'] = mean_discharge + (0.5 * std_discharge)
        df['avg_minus_0.5std'] = mean_discharge - (0.5 * std_discharge)

        #Plotting
        plt.figure(figsize=(10, 7))
        plt.gca().yaxis.grid()

        #Max and Min Years
        plt.plot(df['datetimeUTC'], df[max_year], color='orangered', label=f'Max Discharge Year')
        plt.plot(df['datetimeUTC'], df[min_year], color='blue', label=f'Min Discharge Year')

        #Avg and Standard Deviation
        plt.plot(df['datetimeUTC'], df['avg_discharge'], color='gray', alpha=0)
        plt.plot(df['datetimeUTC'], df['avg_plus_0.5std'], color='gray', alpha=0.8, label='Average +- 0.5 stdev')
        plt.plot(df['datetimeUTC'], df['avg_minus_0.5std'], color='gray', alpha=0.8)

        #Given year's data
        plt.plot(df['datetimeUTC'], df['discharge_10'], color='black', label=f'Original Year')

        #Line to show selected date
        plt.axvline(x=pd.to_datetime(self.date - timedelta(weeks=52)), color='black', linestyle=':')

        plt.fill_between(df['datetimeUTC'], df['avg_plus_0.5std'], df['avg_minus_0.5std'], color='gray', alpha=0.4)

        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

        #Formatting
        plt.suptitle('Trinity River Stream Flow')
        plt.title(f'{self.total_flow(df)} : {self.current_change(df)}', fontsize=10)
        plt.xlabel('Data')
        plt.ylabel('Discharge (CFS)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

if __name__ == '__main__':
    trinity_burnt_ranch_id = '11527000'
    test = Streamflow(trinity_burnt_ranch_id, '2024-11-21')

    curr = test.get_current_year_data()
    curr = test.add_regression_data(curr)

    df = test.merge_data()
    test.graph_stream_data(df)
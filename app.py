import requests
from io import StringIO
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import streamlit as st



data_url= "https://github.com/Ahmedsamy96/Multivariate-Time-Series/blob/main/IOT_temp.csv"

# load data from GitHub
data_response = requests.get(data_url)
df = pd.read_csv(data_response.content)

# Drop the first column since it seems like an index
df.drop(0, axis=1, inplace=True)

# Change column names to understand easily
df.rename(columns={'noted_date':'date', 'out/in':'place'}, inplace=True)

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
df['year'] = df['date'].apply(lambda x : x.year)
df['month'] = df['date'].apply(lambda x : x.month)
df['day'] = df['date'].apply(lambda x : x.day)
df['weekday'] = df['date'].apply(lambda x : x.day_name())
df['weekofyear'] = df['date'].apply(lambda x : x.weekofyear)
df['hour'] = df['date'].apply(lambda x : x.hour)
df['minute'] = df['date'].apply(lambda x : x.minute)

def month2seasons(x):
    if x in [12, 1, 2]:
        season = 'Winter'
    elif x in [3, 4, 5]:
        season = 'Summer'
    elif x in [6, 7, 8, 9]:
        season = 'Monsoon'
    elif x in [10, 11]:
        season = 'Post_Monsoon'
    return season
df['season'] = df['month'].apply(month2seasons)

def hours2timing(x):
    if x in [22,23,0,1,2,3]:
        timing = 'Night'
    elif x in range(4, 12):
        timing = 'Morning'
    elif x in range(12, 17):
        timing = 'Afternoon'
    elif x in range(17, 22):
        timing = 'Evening'
    else:
        timing = 'X'
    return timing
df['timing'] = df['hour'].apply(hours2timing)

df['id'].apply(lambda x : x.split('_')[6]).nunique() == len(df)
df['id'] = df['id'].apply(lambda x : int(x.split('_')[6]))

season_agg = df.groupby('season').agg({'temp': ['min', 'max']})
season_maxmin = pd.merge(season_agg['temp']['max'],season_agg['temp']['min'],right_index=True,left_index=True)
season_maxmin = pd.melt(season_maxmin.reset_index(), ['season']).rename(columns={'season':'Season', 'variable':'Max/Min'})

timing_agg = df.groupby('timing').agg({'temp': ['min', 'max']})
timing_maxmin = pd.merge(timing_agg['temp']['max'],timing_agg['temp']['min'],right_index=True,left_index=True)
timing_maxmin = pd.melt(timing_maxmin.reset_index(), ['timing']).rename(columns={'timing':'Timing', 'variable':'Max/Min'})

tsdf = df.drop_duplicates(subset=['date','place']).sort_values('date').reset_index(drop=True)
tsdf['temp'] = df.groupby(['date','place'])['temp'].mean().values
tsdf.drop('id', axis=1, inplace=True)

in_month = tsdf[tsdf['place']=='In'].groupby('month').agg({'temp':['mean']})
in_month.columns = [f"{i[0]}_{i[1]}" for i in in_month.columns]
out_month = tsdf[tsdf['place']=='Out'].groupby('month').agg({'temp':['mean']})
out_month.columns = [f"{i[0]}_{i[1]}" for i in out_month.columns]

tsdf['daily'] = tsdf['date'].apply(lambda x : pd.to_datetime(x.strftime('%Y-%m-%d')))
in_day = tsdf[tsdf['place']=='In'].groupby(['daily']).agg({'temp':['mean']})
in_day.columns = [f"{i[0]}_{i[1]}" for i in in_day.columns]
out_day = tsdf[tsdf['place']=='Out'].groupby(['daily']).agg({'temp':['mean']})
out_day.columns = [f"{i[0]}_{i[1]}" for i in out_day.columns]

# Extracting data from HoloViews Curves
in_day_data = in_day.reset_index()
out_day_data = out_day.reset_index()

in_wd = tsdf[tsdf['place']=='In'].groupby('weekday').agg({'temp':['mean']})
in_wd.columns = [f"{i[0]}_{i[1]}" for i in in_wd.columns]
in_wd['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in in_wd.index]
in_wd.sort_values('week_num', inplace=True)
in_wd.drop('week_num', axis=1, inplace=True)
out_wd = tsdf[tsdf['place']=='Out'].groupby('weekday').agg({'temp':['mean']})
out_wd.columns = [f"{i[0]}_{i[1]}" for i in out_wd.columns]
out_wd['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in out_wd.index]
out_wd.sort_values('week_num', inplace=True)
out_wd.drop('week_num', axis=1, inplace=True)

# Grouping and aggregating data
in_wof = tsdf[tsdf['place']=='In'].groupby('weekofyear').agg({'temp':'mean'})
out_wof = tsdf[tsdf['place']=='Out'].groupby('weekofyear').agg({'temp':'mean'})

in_tsdf = tsdf[tsdf['place']=='In'].reset_index(drop=True)
in_tsdf.index = in_tsdf['date']

out_tsdf = tsdf[tsdf['place']=='Out'].reset_index(drop=True)
out_tsdf.index = out_tsdf['date']

in_tsdf_int = in_tsdf['temp'].resample('1min').interpolate(method='nearest')
out_tsdf_int = out_tsdf['temp'].resample('1min').interpolate(method='nearest')

inp_df = pd.DataFrame()
in_d_inp = in_day.resample('1D').interpolate('spline', order=5)
out_d_inp = out_day.resample('1D').interpolate('spline', order=5)
inp_df['In'] = in_d_inp.temp_mean
inp_df['Out'] = out_d_inp.temp_mean

org_df = inp_df.reset_index()
org_df['season'] = org_df['daily'].apply(lambda x : month2seasons(x.month))
org_df = pd.get_dummies(org_df, columns=['season'])

def run_prophet(place, prediction_periods, plot_comp=True):
    st.dataframe(df)
    
    # make dataframe for training
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.date_range(start=org_df['daily'][0], end=org_df['daily'][133])
    prophet_df['y'] = org_df[place]
    # add seasonal information
    prophet_df['monsoon'] = org_df['season_Monsoon']
    prophet_df['post_monsoon'] = org_df['season_Post_Monsoon']
    prophet_df['winter'] = org_df['season_Winter']

    # train model by Prophet
    m = Prophet(changepoint_prior_scale=0.1, yearly_seasonality=2, weekly_seasonality=False)
    # include seasonal periodicity into the model
    m.add_seasonality(name='season_monsoon', period=124, fourier_order=5, prior_scale=0.1, condition_name='monsoon')
    m.add_seasonality(name='season_post_monsoon', period=62, fourier_order=5, prior_scale=0.1, condition_name='post_monsoon')
    m.add_seasonality(name='season_winter', period=93, fourier_order=5, prior_scale=0.1, condition_name='winter')
    m.fit(prophet_df)

    # make dataframe for prediction
    future = m.make_future_dataframe(periods=prediction_periods)
    # add seasonal information
    future_season = pd.get_dummies(future['ds'].apply(lambda x : month2seasons(x.month)))
    future['monsoon'] = future_season['Monsoon']
    future['post_monsoon'] = future_season['Monsoon']
    future['winter'] = future_season['Winter']

    # predict the future temperature
    prophe_result = m.predict(future)

    # plot prediction
    fig1 = m.plot(prophe_result)
    ax = fig1.gca()
    ax.set_title(f"{place} Prediction", size=25)
    ax.set_xlabel("Time", size=15)
    ax.set_ylabel("Temperature", size=15)
    st.pyplot(fig1)

    
# Streamlit app
def main():
    st.title("Temperature Prediction App")
    
    # Dropdown for Temperature status
    temp_status = st.selectbox("Temperature status:", ["IN", "OUT"])
    
    # Number input for TimePoints
    time_points = st.number_input("TimePoints:", min_value=1, step=1, value=30)
    
    # Button to execute the code
    if st.button("Run Prophet"):
        if temp_status == "IN":
            run_prophet("In", time_points)
        else:
            run_prophet("Out", time_points)

if __name__ == "__main__":
    main()

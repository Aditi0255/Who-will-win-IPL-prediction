# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:50:22 2022

@author: ADITI
"""

import streamlit as stl
import pickle
import pandas as pd

stl.title("WHO WILL WIN - Ipl prediction")
current_teams = ['Chennai Super Kings','Sunrisers Hyderabad','Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kings XI Punjab','Rajasthan Royals','Delhi Capitals','Kolkata Knight Riders']

destination = ['Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack''Hyderabad', 'Sharjah', 'Mohali', 'Bengaluru','Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       ]

new1 = pickle.load(open('pipe.pkl','rb'))
destination1 = stl.selectbox('Select playing destination',sorted(destination))
chasing_score = stl.number_input('Chasing Score')

team1 = stl.selectbox('Chasing team',sorted(current_teams))

team2 = stl.selectbox('Defending team',sorted(current_teams))

overs = stl.number_input('Overs completed')

current_score = stl.number_input('Current Score')
wickets = stl.number_input('Wickets Down')

if stl.button('Predict the result'):
    remaining_runs = chasing_score - current_score
    remaining_balls = 120 - (overs*6)
    wickets = 10 - wickets
    crr = current_score/overs
    rrr = (remaining_runs*6)/remaining_balls

    input_df = pd.DataFrame({'batting_team':[team1],'bowling_team':[team2],'city':[destination1],'remaining runs':[remaining_runs],'remaining balls':[remaining_balls],'wickets left':[wickets],'total_runs_x':[chasing_score],'Crr':[crr],'Rrr':[rrr]})

    result = new1.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    stl.header("Chances of winning of"+team1 + "- " + str(round(win*100)) + "%")
    stl.header("Chances of winning of"+team2 + "- " + str(round(loss*100)) + "%")

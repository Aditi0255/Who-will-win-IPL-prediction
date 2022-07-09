# -*- coding: utf-8 -*-
"""

@author: ADITI
"""

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

matches = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')
def who_won(x):
    if (x['batting_team'] == x['winner']):
        return 1
    else:
        return 0

score= delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
score = score[score['inning'] == 1]
score['total_runs']=score['total_runs']+1
game = matches.merge(score[['match_id','total_runs']],left_on='id',right_on='match_id')
current_teams = ['Chennai Super Kings','Sunrisers Hyderabad','Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kings XI Punjab','Rajasthan Royals','Delhi Capitals','Kolkata Knight Riders']
game['team1'] = game['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
game['team2'] = game['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
game['team1'] = game['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
game['team2'] = game['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

game = game[game['team1'].isin(current_teams)]
game =game[game['team2'].isin(current_teams)]

game = game[game['dl_applied'] == 0]
game = game[['match_id','city','winner','total_runs']]
score = game.merge(delivery,on='match_id')
score= score[score['inning'] == 2]
score['current score'] = score.groupby('match_id').cumsum()['total_runs_y']
score['remaining runs'] = score['total_runs_x'] - score['current score']
score['remaining balls']= 126 - (score['over']*6 + score['ball'])
score['player_dismissed'] = score['player_dismissed'].fillna("0")
score['player_dismissed'] = score['player_dismissed'].apply(lambda x:x if x == "0" else "1")
score['player_dismissed'] = score['player_dismissed'].astype('int')
wickets = score.groupby('match_id').cumsum()['player_dismissed'].values
score['wickets left'] = 10 - wickets
score['Crr'] = (score['current score']*6)/(120 - score['remaining balls'])
score['Rrr'] = (score['remaining runs']*6)/score['remaining balls']
score['result'] = score.apply(who_won,axis=1)
required_data = score[['batting_team','bowling_team','city','remaining runs','remaining balls','wickets left','total_runs_x','Crr','Rrr','result']]
required_data = required_data.sample(required_data.shape[0])
required_data.dropna(inplace=True)
required_data = required_data[required_data['remaining balls'] != 0]

X =required_data.iloc[:,:-1]
y =required_data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


new= ColumnTransformer([
    ('new',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
],remainder='passthrough')

new1 = Pipeline(steps=[
    ('step1',new),
    ('step2',LogisticRegression(solver='liblinear'))])
new1.fit(X_train,y_train)
output = new1.predict(X_test)

accuracy=accuracy_score(y_test,output)
pickle.dump(new1,open('pipe.pkl','wb'))

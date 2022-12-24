import streamlit as st
from Home import get_data
import pandas as pd
import requests
import json
import numpy as np
import time
import plotly.express as px  # pip install plotly-express

st.set_page_config(
    page_title="Human Activity Classification", 
    page_icon=":running:", 
    layout="wide"
    )

X_test = np.load('test.npy')

np.shape(X_test)

emojis_dict = {'Downstairs' : ' :arrow_down: ',
               'Jogging' :  ' :running: ',
               'Sitting' : ' :seat: ',
               'Standing' :' :octocat: ',
               'Upstairs' : ' :arrow_upper_right: ' ,
               'Walking' : ' :feet: '
                   }

#TODO animate plotly plots 
#TODO When time found check deployment with docker and all.
if st.button("Start Predicting"):
    if st.button("Stop Predicting"):
            st.stop()
    pl1 = st.empty() 
    pl2 = st.empty() 
    for i in range(X_test.shape[0]):
        
        first = X_test[i].reshape(80)
        first = first.tolist()
        test_df = pd.DataFrame()
        test_df['timestamp'] = first[0::4]
        test_df['x-axis'] = first[1::4]
        test_df['y-axis'] = first[2::4]
        test_df['z-axis'] = first[3::4]
        test_df.plot(x = 'timestamp')
        
        for j in range(test_df.shape[0]):
            fig = px.line(test_df[:j], x="timestamp", y = ['x-axis' , 'y-axis', 'z-axis'] )

            fig.update_layout(title='',
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=(dict(showgrid=False)),
                            )

            pl1.plotly_chart(fig, use_container_width=True)
            
        inputs = {
            "user_input" : list(X_test[i].reshape(80))
            }
        pl2.info('Predicting...')
        res = requests.post(url = "http://127.0.0.1:8000/predict",  data = json.dumps(inputs))
        pred = res.json()
        pl2.info(f" {emojis_dict[pred.get('Prediction')[0]]}  {pred.get('Prediction')[0]}")
        time.sleep(5)
        


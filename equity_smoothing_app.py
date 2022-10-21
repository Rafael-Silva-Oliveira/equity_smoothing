# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.signal import savgol_filter
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import base64
from PIL import Image
import pendulum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.set_page_config(layout="wide")

    
#equity = pd.read_excel(r'C:\Users\RafaelOliveira\OneDrive - Brand Delta\Documents\Projects\new_data_science\data_science\data_science\equity_scores\equity_values_green_cuisine_total.xlsx')

json_file = json.load(open(r'C:\Users\RafaelOliveira\OneDrive - Brand Delta\Documents\Projects\new_data_science\data_science\data_science\equity_scores\config_file.json'))
old_file = json_file["old_tagged_file"]
new_file = json_file["refreshed_data_file"]
save_clean_file = json_file["new_tagged_file_path"]
refresh = json_file['data_refresh']
time_periods = json_file['time_periods']
market = json_file['market']
brands = json_file['brands']
aw_wt = json_file['weights']['awareness']
sa_wt = json_file['weights']['saliency']
af_wt = json_file['weights']['affinity']
smoothening = json_file['smoothening']
old_equity_file_path = json_file['old_equity_file_path']
new_equity_file_path = json_file['new_equity_file_path']
campaings = json_file['campaings']
campaings_equity_calculations = json_file['campaings_equity_calculations_only']
periodic_equity_calculations = json_file['periodic_equity_calculations_only']
gtrends_path = json_file['google_trends_path']
models_path = json_file['models_path']


def _smoothing(equity, col, smooth_filter, window, order, degree,brands = {'meat_alternatives':["green_cuisine", "Quorn", "Linda_McCartney"]}):
    """Smoothing

    Function to smooth the data

    Args:
        equity (dataframe): Equity scores pre-smooth
        col (str): Name of the column of the time period
        window (int): Window value for the smoothing

    Returns:
        moving_average (dataframe): Smoothed values 
        unnormalised (dataframe): 
        long_form (dataframe): 

    Raises:
        AnyError: If anything bad happens.
    """

    #create list of KPI metrics to running smoothening on
    metrics = list(equity.columns)
    metrics.remove(col)
    metrics.remove('Brand')
    if col == 'Week_Commencing':
        equity[col] = pd.to_datetime(equity[col])
        equity[col] = equity[col].apply(lambda x: x.strftime('%Y-%m-%d'))
    structure = equity[[col]].drop_duplicates()
    ws = window

    if smooth_filter == 'rolling_mean':
        moving_average = pd.DataFrame()
        for brand in equity['Brand'].unique():
            brand_cut = equity[equity['Brand']==brand].sort_values(by=[col])
            #join to the master structure to address when we have missing data
            brand_cut = pd.merge(structure,brand_cut,on=[col],how='left')
            #populate missing data
            brand_cut['Brand'] = brand
            brand_cut = brand_cut.fillna(0)
            #iterate through the KPI metrics
            for kpi in metrics:
                #compute rolling average for each of the metrics
                variable_name = str(kpi) 
                brand_cut[variable_name] = brand_cut[kpi].\
                    rolling(window=int(ws)).mean()
            #join the brand processed data to the master

            moving_average = pd.concat([moving_average, brand_cut],
                                       axis=0, ignore_index=True)
            moving_average.dropna(inplace=True)
            moving_average = moving_average.astype({"Reach": int, col: str})
            moving_average = moving_average.\
                sort_values(by=[col], ascending=True).reset_index(drop='True')
            #save and unnormalised master version
            unnormalised = moving_average.copy('deep').fillna(0)
    if smooth_filter == 'savgol':
        moving_average = pd.DataFrame()
        for brand in equity['Brand'].unique():
            brand_cut = equity[equity['Brand']==brand].sort_values(by=[col])
            #join to the master structure to address when we have missing data
            brand_cut = pd.merge(structure,brand_cut,on=[col],how='left')
            #populate missing data
            brand_cut['Brand'] = brand
            brand_cut = brand_cut.fillna(0)
            #iterate through the KPI metrics
            for kpi in metrics:
                #compute rolling average for each of the metrics
                variable_name = str(kpi) 
                brand_cut[variable_name] = savgol_filter(brand_cut[kpi],
                                                         window_length = window,
                                                         polyorder = order)
            #join the brand processed data to the master
            moving_average = pd.concat([moving_average, brand_cut],
                                       axis=0, ignore_index=True)
            moving_average = moving_average.astype({"Reach": int, col: str})
            moving_average = moving_average.\
                sort_values(by=[col],ascending=True).reset_index(drop='True')
            #save and unnormalised master version
            unnormalised = moving_average.copy('deep').fillna(0)

    if smooth_filter == 'polynomial':
        moving_average = pd.DataFrame()
        for brand in equity['Brand'].unique():
            brand_cut = equity[equity['Brand']==brand].sort_values(by=[col])
            #join to the master structure to address when we have missing data
            brand_cut = pd.merge(structure,brand_cut,on=[col],how='left')
            #populate missing data
            brand_cut['Brand'] = brand
            brand_cut = brand_cut.fillna(0)
            #iterate through the KPI metrics
            for kpi in metrics:
                #compute rolling average for each of the metrics
                variable_name = str(kpi) 
                poly = np.polyfit(brand_cut.index, brand_cut[kpi], degree)
                brand_cut[variable_name] = np.poly1d(poly)(brand_cut.index)
            #join the brand processed data to the master
            moving_average = pd.concat([moving_average, brand_cut],
                                       axis=0, ignore_index=True)
            moving_average = moving_average.astype({"Reach": int, col: str})
            moving_average = moving_average.\
                sort_values(by=[col], ascending=True).reset_index(drop='True')
            #save and unnormalised master version
            unnormalised = moving_average.copy('deep').fillna(0)

    all_columns = list(moving_average.columns)
    melt_columns = [name for name in all_columns if name not in [col, 'Brand']]
    long_form = pd.melt(moving_average , id_vars=[col, 'Brand'],
                        value_vars=melt_columns)
    return moving_average, unnormalised, long_form


def extra_equity(equity, smooth_filter,smooth_checkbox,window, order, degree,brands = {'meat_alternatives':["green_cuisine", "Quorn", "Linda_McCartney"]},index_brand = 'green_cuisine', col='Week_Commencing',brand='green_cuisine',cut=False,
averaged_values=True, smooth_values=True):

    if smooth_values:
        if col == ('Week_Commencing') or (col=='Month_Commencing'):
            if smooth_checkbox:
                smooth_equity, unnormalised, long_form = _smoothing(equity, col,
                                                                    smooth_filter,
                                                                    window, order,
                                                                    degree)
            elif smooth_checkbox == False:
                unnormalised = equity
            data = unnormalised.copy('deep').fillna(0)
            awareness_metrics = ['eSoV', 'Reach']
            saliency_metrics = ['Average Engagement',
                                'Usage SoV',
                                'Search_Index']
            affinity_metrics = ['Star - Senses',
                                'Star - Nutrition',
                                'Star - Sustainability',
                                'Star - Functionality',
                                'Star - Brand Strength']
            data  = data[[col, 'Brand', 'eSoV', 'Reach',
                    'Average Engagement', 'Usage SoV',
                    'Search_Index', 'Star - Senses',
                    'Star - Nutrition', 'Star - Sustainability',
                    'Star - Functionality',
                    'Star - Brand Strength']]
    #     else:
    #         data = equity
    #         awareness_metrics = ['eSoV', 'Reach']
    #         saliency_metrics = ['Average Engagement',
    #                             'Usage SoV',
    #                             'Search_Index']
    #         affinity_metrics = ['Star - Senses',
    #                             'Star - Nutrition',
    #                             'Star - Sustainability',
    #                             'Star - Functionality',
    #                             'Star - Brand Strength']
    # else:
    #     data = equity
    #     awareness_metrics = ['eSoV', 'Reach']
    #     saliency_metrics = ['Average Engagement',
    #                         'Usage SoV',
    #                         'Search_Index']
    #     affinity_metrics = ['Star - Senses',
    #                         'Star - Nutrition',
    #                         'Star - Sustainability',
    #                         'Star - Functionality',
    #                         'Star - Brand Strength']
    #Compute normalised indexed performance for lower-level metrics
    all_equity_metrics = awareness_metrics+saliency_metrics+affinity_metrics
    #Compute the proportioned performance for the lower-level metrics
    data[all_equity_metrics] = data[all_equity_metrics].clip(lower=0)
    brand_list = brands[list(brands.keys())[0]]
    data = data[data['Brand'].isin(brand_list)]
    #return data[data['Brand'] == brand]
    metrics_df_list = []
    metrics_df = pd.DataFrame()

    for metric in all_equity_metrics:
        debug_test_1 = data[(data['Brand']==index_brand)&(~data[metric].isnull())\
                          &(data[metric] > 0)][metric].reset_index(drop=True)
        if debug_test_1.empty:
            index=1
        else:
            index = data[(data['Brand']==index_brand)&(~data[metric].isnull())\
                         &(data[metric] > 0)][metric].reset_index(drop=True)[0]
        indexed_data = data[[col, 'Brand', metric]]
        indexed_data[metric] = indexed_data[metric]/index
        # proportion the performance of the baselined values normalised
        proportioned_index = (indexed_data.groupby([col, 'Brand'])[metric]\
                              .sum()/indexed_data.groupby([col])[metric]\
                                  .sum()).reset_index()
        if len(metrics_df) == 0:
            metrics_df = proportioned_index.copy('deep')
        else:
            metrics_df = pd.merge(metrics_df, proportioned_index, on=[col,'Brand'], how='left')

    metrics_df_list.append(metrics_df)
    metrics_df_final = pd.concat(metrics_df_list)
    #Compute the framework level scores
    metrics_df_final['Framework - Awareness'] = \
        metrics_df_final[awareness_metrics].mul(aw_wt).sum(axis=1)*100
    metrics_df_final['Framework - Saliency'] = \
        metrics_df_final[saliency_metrics].mul(sa_wt).sum(axis=1)*100
    metrics_df_final['Framework - Affinity'] = \
        metrics_df_final[affinity_metrics].mul(af_wt).sum(axis=1)*100

    if averaged_values:
        metrics_df_final = metrics_df_final
    else:
        metrics_df_final = metrics_df_final.sort_values([col,'Brand']).reset_index(drop=True)
        metrics_df_final  = pd.concat([data.sort_values([col,'Brand']).reset_index(drop=True),
                                        metrics_df_final[['Framework - Awareness',
                                                          'Framework - Saliency',
                                                          'Framework - Affinity']]],axis=1)
    try:
        metrics_df_final = metrics_df_final[metrics_df_final['Brand'] \
                                            == brand].fillna(0)
    except TypeError:
        metrics_df_final[col] = metrics_df_final[col].astype(str)
        metrics_df_final = metrics_df_final[metrics_df_final['Brand'] \
                                            == brand].fillna(0)
    return metrics_df_final



########################
### ANALYSIS METHODS ###
########################
def affinity_spend(equity,brand='green_cuisine'):

    affinity_brand_mapping = {
        'green_cuisine':'Green Cuisine'
    }
    equity_cp = equity.copy()
    equity_cp['Brand'] = equity_cp['Brand'].map(affinity_brand_mapping)
    brand = affinity_brand_mapping[brand]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=equity_cp['Week_Commencing'], 
            y=equity_cp['Framework - Awareness']
            ,name='Awareness'),
            secondary_y=True)
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=equity_cp['Week_Commencing'], 
            y=equity_cp['Framework - Saliency'],
            name='Saliency'),
            secondary_y=True)
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=equity_cp['Week_Commencing'], 
            y=equity_cp['Framework - Affinity'],
            name='Affinity'),
            secondary_y=True)
    fig.update_layout(
        #margin=dict(l=2, r=1, t=55, b=2),
        autosize=True,
        xaxis=dict(title_text="Time"),
        width=1000,title="{} Equity Smoothing Experiment".format(brand),
        )

    # fig.add_annotation(x=0, y=1.1, xref='paper', yref='paper', text='             Pre             ', showarcolumn=False, font=dict(color='white'), bgcolor='blue')
    # fig.add_annotation(x=0.31, y=1.1, xref='paper', yref='paper', text='            During            ', showarcolumn=False, font=dict(color='blue'), bgcolor='gray')
    # fig.add_annotation(x=0.94, y=1.1, xref='paper', yref='paper', text='            Post            ', showarcolumn=False, font=dict(color='blue'), bgcolor='green')
    st.plotly_chart(fig)

uploaded_files = st.file_uploader('Upload the incomplete equity file',type='xlsx', accept_multiple_files=True)

with st.sidebar:
    smoothing_filter = st.selectbox("Which smoothing filter would you like to use?",['rolling_mean','savgol','polynomial'])

    window = st.slider("Which window would you like to use?",2,32,1)

    order = st.slider("Which order would you like to use?",2,32,1)

    degree = st.slider("Which degree would you like to use?",2,32,1)

    smoothing_checkbox = st.checkbox("Would you like to plot with smoothing or no smoothing?",value=True)

option = st.selectbox(
        'Would you like to use total or averaged equities?',
        ('Total','Average'))

if option == "Total":
    file = [f for f in uploaded_files if "total" in f.name]
    equity = pd.read_excel(file[0])

if option == "Average":
    file = [f for f in uploaded_files if "average" in f.name]
    equity = pd.read_excel(file[0])

with st.expander("Click to see dataframe of the {} equities".format(option)):
    st.dataframe(equity)

equity_v2 = extra_equity(equity, smoothing_filter,smoothing_checkbox, window,order, degree)
with st.expander("Click to see dataframe with the {} framework metrics".format(option)):
    st.write(equity_v2)
affinity_spend(equity_v2,'green_cuisine')

st.info("In order to plot the chart, please upload the equity files first")

#, "monthly", "quaterly", "yearly", "YTD"


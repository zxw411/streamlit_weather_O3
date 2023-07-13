import json
import time
import random
import datetime

import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_echarts import st_echarts

from streamlit.server.server import Server
# from streamlit.script_run_context import get_script_run_ctx as get_report_ctx
from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx

import graphviz
import pydeck as pdk
import altair as alt
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from pyecharts.charts import *
from pyecharts.globals import ThemeType
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode

from PIL import Image
from io import BytesIO

import torch
from sklearn import preprocessing
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import datetime
import joblib

def main():
    st.set_page_config(page_title="Leice-zxw",page_icon=":rainbow:",layout="wide",initial_sidebar_state="auto")
    st.title('天气预报')
    st.markdown('<br>',unsafe_allow_html=True)
    st.markdown('<br>',unsafe_allow_html=True)
    charts_mapping={
        'Line':'line_chart','Bar':'bar_chart','Area':'area_chart','Hist':'pyplot','Altair':'altair_chart',
        'Map':'map','Distplot':'plotly_chart','Pdk':'pydeck_chart','Graphviz':'graphviz_chart','PyEchart':''
    }
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit=True
    else:
        st.session_state.first_visit=False
    # 初始化全局配置
    if st.session_state.first_visit:
        # 在这里可以定义任意多个全局变量，方便程序进行调用
        st.session_state.date_time=datetime.datetime.now() + datetime.timedelta(hours=8) # Streamlit Cloud的时区是UTC，加8小时即北京时间
        st.session_state.random_chart_index=random.choice(range(len(charts_mapping)))
        st.session_state.my_random=MyRandom(random.randint(1,1000000))
        st.session_state.city_mapping,st.session_state.random_city_index=get_city_mapping()
        # st.session_state.random_city_index=random.choice(range(len(st.session_state.city_mapping)))
        st.balloons()
        st.snow()

    d=st.sidebar.date_input('Date',st.session_state.date_time.date())
    t=st.sidebar.time_input('Time',st.session_state.date_time.time())
    t=f'{t}'.split('.')[0]
    st.sidebar.write(f'The current date time is {d} {t}')
   # chart=st.sidebar.selectbox('Select Chart You Like',charts_mapping.keys(),index=st.session_state.random_chart_index)
    city=st.sidebar.selectbox('Select City You Like',st.session_state.city_mapping.keys(),index=st.session_state.random_city_index)

    with st.container():
        st.markdown(f'### {city} Weather Forecast')
        forecastToday,df_forecastHours,df_forecastDays=get_city_weather(st.session_state.city_mapping[city])
        col1,col2,col3,col4,col5,col6=st.columns(6)
        col1.metric('Weather',forecastToday['weather'])
        col2.metric('Temperature',forecastToday['temp'])
        col3.metric('Body Temperature',forecastToday['realFeel'])
        col4.metric('Humidity',forecastToday['humidity'])
        col5.metric('Wind',forecastToday['wind'])
        col6.metric('UpdateTime',forecastToday['updateTime'])

        c1 = (
            Line()
            .add_xaxis(df_forecastHours.index.to_list())
            .add_yaxis('Temperature', df_forecastHours.Temperature.values.tolist())
            .add_yaxis('Body Temperature', df_forecastHours['Body Temperature'].values.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title="24 Hours Forecast"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value} °C")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1] + '°C';}")))
        )

        c2 = (
            Line()
            .add_xaxis(xaxis_data=df_forecastDays.index.to_list())
            .add_yaxis(series_name="High Temperature",y_axis=df_forecastDays.Temperature.apply(lambda x:int(x.replace('°C','').split('~')[1])))
            .add_yaxis(series_name="Low Temperature",y_axis=df_forecastDays.Temperature.apply(lambda x:int(x.replace('°C','').split('~')[0])))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="7 Days Forecast"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value} °C")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
                )
            .set_series_opts(label_opts=opts.LabelOpts(formatter=JsCode("function(x){return x.data[1] + '°C';}")))
        )

        t = Timeline(init_opts=opts.InitOpts(theme=ThemeType.LIGHT,width='1200px'))
        t.add_schema(play_interval=10000,is_auto_play=True)
        t.add(c1, "24 Hours Forecast")
        t.add(c2, "7 Days Forecast")
        components.html(t.render_embed(), width=1200, height=520)
        st.markdown(f'### 基于LSTM的平度臭氧短时预报')
        date,ans_O3= get_O3(st.session_state.date_time)
        O3 = (
            Line()
            .add_xaxis(date)
            .add_yaxis('O3', ans_O3)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="O3"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value",axislabel_opts=opts.LabelOpts(formatter="{value}")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )

        components.html(O3.render_embed(), width=1200, height=520)

        with st.expander("24 Hours Forecast Data"):
            st.table(df_forecastHours.style.format({'Temperature':'{}°C','Body Temperature':'{}°C','Humidity':'{}%'}))
        with st.expander("7 Days Forecast Data",expanded=True):
            st.table(df_forecastDays)


class MyRandom:
    def __init__(self,num):
        self.random_num=num

def my_hash_func(my_random):
    num = my_random.random_num
    return num


@st.cache(ttl=3600)
def get_city_mapping():
    url='https://h5ctywhr.api.moji.com/weatherthird/cityList'
    r=requests.get(url)
    data=r.json()
    city_mapping=dict()
    guangzhou=-1
    flag=True
    for i in data.values():
        for each in i:
            city_mapping[each['name']]=each['cityId']
            if each['name'] != '青岛市' and flag:
                guangzhou+=1
            else:
                flag=False
    return city_mapping,guangzhou

@st.cache(ttl=3600)
def get_city_weather(cityId):
    url='https://h5ctywhr.api.moji.com/weatherDetail'
    headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
    data={"cityId":cityId,"cityType":0}
    r=requests.post(url,headers=headers,json=data)
    result=r.json()

    # today forecast
    forecastToday=dict(
        humidity=f"{result['condition']['humidity']}%",
        temp=f"{result['condition']['temp']}°C",
        realFeel=f"{result['condition']['realFeel']}°C",
        weather=result['condition']['weather'],
        wind=f"{result['condition']['windDir']}{result['condition']['windLevel']}级",
        updateTime=(datetime.datetime.fromtimestamp(result['condition']['updateTime'])+datetime.timedelta(hours=8)).strftime('%H:%M:%S')
    )

    # 24 hours forecast
    forecastHours=[]
    for i in result['forecastHours']['forecastHour']:
        tmp={}
        tmp['PredictTime']=(datetime.datetime.fromtimestamp(i['predictTime'])+datetime.timedelta(hours=8)).strftime('%H:%M')
        tmp['Temperature']=i['temp']
        tmp['Body Temperature']=i['realFeel']
        tmp['Humidity']=i['humidity']
        tmp['Weather']=i['weather']
        tmp['Wind']=f"{i['windDesc']}{i['windLevel']}级"
        forecastHours.append(tmp)
    df_forecastHours=pd.DataFrame(forecastHours).set_index('PredictTime')

    # 7 days forecast
    forecastDays=[]
    day_format={1:'昨天',0:'今天',-1:'明天',-2:'后天'}
    for i in result['forecastDays']['forecastDay']:
        tmp={}
        now=datetime.datetime.fromtimestamp(i['predictDate'])+datetime.timedelta(hours=8)
        diff=(st.session_state.date_time-now).days
        festival=i['festival']
        tmp['PredictDate']=(day_format[diff] if diff in day_format else now.strftime('%m/%d')) + (f' {festival}' if festival != '' else '')
        tmp['Temperature']=f"{i['tempLow']}~{i['tempHigh']}°C"
        tmp['Humidity']=f"{i['humidity']}%"
        tmp['WeatherDay']=i['weatherDay']
        tmp['WeatherNight']=i['weatherNight']
        tmp['WindDay']=f"{i['windDirDay']}{i['windLevelDay']}级"
        tmp['WindNight']=f"{i['windDirNight']}{i['windLevelNight']}级"
        forecastDays.append(tmp)
    df_forecastDays=pd.DataFrame(forecastDays).set_index('PredictDate')
    return forecastToday,df_forecastHours,df_forecastDays


@st.cache(ttl=3600)
def get_O3(now):
    model1 = torch.load(r"./model/model.pkl",map_location=torch.device('cpu'))
    model2 = torch.load(r"model/model_2.pkl", map_location=torch.device('cpu'))
    model3 = torch.load(r"./model/model_3.pkl", map_location=torch.device('cpu'))
    data = pd.read_csv(r"./data/test.csv")
    num = 0
    for i in range(0, len(data)):
        data_time = datetime.datetime.strptime(data['date'][i], "%Y/%m/%d %H:%M")
        if now.month == data_time.month and now.day == data_time.day and now.hour == data_time.hour:
            num = i
            break
    ans_O3 =data[num - 23+3:num + 1]["O3"].to_list()
    ans_date = []
    for i in data[num - 23+3:num + 1+3]["date"].values:
        ans_date.append(i.split(" ")[1])
    test1 = data[num - 23:num + 1].interpolate(method='linear').reset_index(drop=True)
    test1 = test1.drop(["date"], axis=1)
    cols = test1.columns.tolist()
    cols = cols[-11:] + cols[:-11]
    test1 = test1[cols]
    test_x_mm = joblib.load(r"./model/test_x")
    test_y_mm = joblib.load(r"./model/test_y")
    X = test_x_mm.transform(np.array(test1)).reshape(1, 24, 14)
    X = torch.from_numpy(X).to(torch.float32).to(device)
    model1.eval()
    with torch.no_grad():
        y_test_pred1 = model1(X)
    ans_O3.append(int(test_y_mm.inverse_transform(y_test_pred1.cpu())))
    model2.eval()
    with torch.no_grad():
        y_test_pred2 = model2(X)
    ans_O3.append(int(test_y_mm.inverse_transform(y_test_pred2.cpu())))
    model3.eval()
    with torch.no_grad():
        y_test_pred3 = model3(X)
    ans_O3.append(int(test_y_mm.inverse_transform(y_test_pred3.cpu())))
    return ans_date,ans_O3

if __name__ == '__main__':
    class LSTM(nn.Module):
        def __init__(self, feature_size, hidden_size, num_layers, output_size):
            super(LSTM, self).__init__()
            self.feature_size = feature_size
            self.hidden_size = hidden_size  # 隐层大小
            self.num_layers = num_layers  # lstm层数
            self.num_directions = 1
            # feature_size为特征维度，就是每个时间点对应的特征数量，这里为6
            self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(self.num_directions * hidden_size, output_size)

        def forward(self, x, hidden=None):
            batch_size = x.shape[0]  # 获取批次大小

            # 初始化隐层状态
            if hidden is None:
                h_0 = x.data.new(self.num_directions * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
                c_0 = x.data.new(self.num_directions * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            else:
                h_0, c_0 = hidden

            # LSTM运算
            output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

            # 全连接层
            output = torch.sigmoid(self.fc(output))  # 形状为batch_size * timestep, 1

            # 我们只需要返回最后一个时间片的数据即可
            return output[:, -1, :]
    main()
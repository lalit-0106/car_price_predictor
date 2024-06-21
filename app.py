import streamlit as st
import pickle
import sklearn
import numpy as np


#Import thr model
model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Predictor')

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

#Ram
ram = st.selectbox('RAM (in GB)', df['Ram'].unique())

#weight
weight = st.number_input('Weight of the Laptop')

#TouchScree
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

#IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

#screen_size
screen_size = st.number_input(label='Screen Size')

#resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#Cpu
cpu = st.selectbox('CPU Brand', df['CPU Brand'].unique())

#hdd
hdd = st.selectbox('HDD (in GB)', [   0,  500, 1000, 2000,   32,  128])

#ssd
ssd = st.selectbox('SSD (in GB)', df['SSD'].unique())

#os
os = st.selectbox('OS', df['OS'].unique())

#gpu
gpu = st.selectbox('GPU', df['GPU Brand'].unique())

if st.button('Predict Price'):
    #query

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])

    ppi = pow((pow(x_res, 2) + pow(y_res, 2)),0.5)/(screen_size)
    query = np.array([ram, weight, 0 if touchscreen == 'No' else 1, 0 if 'No' else 1,
                      ppi, ssd, hdd, 1 if company == 'Acer' else 0, 1 if company == 'Apple' else 0,
                      1 if company == 'Asus' else 0, 1 if company == 'Chuwi' else 0,
                      1 if company == 'Dell' else 0, 1 if company == 'Fujitsu' else 0,
                      1 if company == 'Google' else 0, 1 if company == 'HP' else 0,
                      1 if company == 'Huawei' else 0, 1 if company == 'LG' else 0, 
                      1 if company == 'Lenovo' else 0, 1 if company == 'MSI' else 0,
                      1 if company == 'Mediacom' else 0, 1 if company == 'Microsoft' else 0,
                      1 if company == 'Razer' else 0, 1 if company == 'Samsung' else 0,
                      1 if company == 'Toshiba' else 0, 1 if company == 'Vero' else 0,
                      1 if company == 'Xiaomi' else 0, 1 if type == '2 in 1 Convertible'else 0,
                      1 if type == 'Gaming' else 0, 1 if type == 'Netbook' else 0,
                      1 if type == 'Notebook' else 0, 1 if type == 'Ultrabook' else 0,
                      1 if type == 'Workstation' else 0, 1 if cpu == 'AMD Processor' else 0,
                      1 if cpu == 'Intel Core i3' else 0, 1 if cpu == 'Intel Core i5' else 0,
                      1 if cpu == 'Intel Core i7' else 0, 1 if cpu == 'Other Intel Processor' else 0,
                      1 if gpu == 'AMD' else 0, 1 if gpu == 'Intel' else 0,
                      1 if gpu == 'Nvidia' else 0, 1 if os == 'Mac' else 0,
                      1 if os == 'Others/No OS/Linux' else 0, 1 if os == 'Windows' else 0])
    
    query = query.reshape(1, 43)
    st.title(f'The Predicted price of the above configuration is : {int(np.exp(model.predict(query)[0]))}')
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./data/Sleep_health_and_lifestyle_dataset.csv')

html_0 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h3>Example data table</h3></center>
</div>
"""
st.markdown(html_0, unsafe_allow_html=True)
st.markdown("")
st.write(df.head(10))

html_1 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h3>The relationship between (sex) and (Sleep Disorder)</h3></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

grouped_data = df.groupby(['Sleep Disorder', 'Gender']).size().reset_index(name='count')
pivot_table = pd.pivot_table(grouped_data, values='count', index='Sleep Disorder', columns='Gender', fill_value=0)
fig, ax = plt.subplots(figsize=(15, 7))
colors = ['#3085C3', '#FF9EAA', '#5CD2E6']
pivot_table.plot.pie(subplots=True, autopct='%1.1f%%', ax=ax, colors=colors)
plt.axis('equal')
st.pyplot(fig)


html_2 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h3>The relationship between (Age) and (Sleep Disorder)</h3></center>
</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
st.markdown("")



group_age = df.groupby(['Age', 'Sleep Disorder']).size().reset_index(name='count')
st.bar_chart(group_age, x='Age', y='count', color='Sleep Disorder', height=400)

html_3 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h3>The relationship between (BMI) and (Sleep Disorder)</h3></center>
</div>
"""
st.markdown(html_3, unsafe_allow_html=True)
st.markdown("")

pivot_table = df.pivot_table(index='BMI Category', columns='Sleep Disorder', aggfunc={'Sleep Disorder': 'count'})
fig, ax = plt.subplots(figsize=(20, 10))
pivot_table.plot.pie(subplots=True, autopct='%1.1f%%', ax=ax, colors=['#FF9EAA', '#FFD0D0', '#5CD2E6', '#3085C3'])
plt.axis('equal') 
st.pyplot(fig)

html_4 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h3>The relationship between (Occupation) and (Sleep Disorder)</h3></center>
</div>
"""
st.markdown(html_4, unsafe_allow_html=True)
st.markdown("")
group_age = df.groupby(['Occupation', 'Sleep Disorder']).size().reset_index(name='count')
st.bar_chart(group_age, x='Occupation', y='count', color='Sleep Disorder', height=400)
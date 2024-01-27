import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

html_1 = """
<div style="background-color:#0E1117;margin-top:40px;padding:5px;border-radius:5px;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h4>การวิเคราะห์และทำนายความเสี่ยงในการเสียชีวิตจากภาวะหัวใจล้มเหลวของผู้ป่วยโรคหัวใจและหลอดเลือดที่อายุมากกว่า50ปี</h4><h5>
Analysis and Prediction of Mortality Risk from Heart Failure in Patients with Cardiovascular Disease Aged Over 50 Years</h5></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

col1, col2, col3 = st.columns([2.5, 6, 1])

with col1:
    st.write("") 

with col2:
    st.image("./pic/heartmaivi.jpg")

with col3:
    st.write("")

### Visualization ###
df = pd.read_excel('./data/heart_failure.xlsx')

# html_3 = """
# <div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
# <center><h3>Example data table</h3></center>
# </div>
# """
# st.markdown(html_3, unsafe_allow_html=True)
# st.markdown("")
# st.write(df.head(10))

# html_4 = """
# <div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
# <center><h3>Count plot for various categorical features</h3></center>
# </div>
# """
# st.markdown(html_4, unsafe_allow_html=True)
# st.markdown("")

# fig = plt.figure(figsize=(18, 15))
# gs = fig.add_gridspec(3, 3)
# gs.update(wspace=0.5, hspace=0.25)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[0, 2])
# ax3 = fig.add_subplot(gs[1, 0])
# ax4 = fig.add_subplot(gs[1, 1])
# ax5 = fig.add_subplot(gs[1, 2])
# ax6 = fig.add_subplot(gs[2, 0])
# ax7 = fig.add_subplot(gs[2, 1])
# ax8 = fig.add_subplot(gs[2, 2])

# background_color = "#ffe6e6"
# color_palette = ["#800000", "#8000ff", "#6aac90", "#5833ff", "#da8829"]
# fig.patch.set_facecolor(background_color)
# ax0.set_facecolor(background_color)
# ax1.set_facecolor(background_color)
# ax2.set_facecolor(background_color)
# ax3.set_facecolor(background_color)
# ax4.set_facecolor(background_color)
# ax5.set_facecolor(background_color)
# ax6.set_facecolor(background_color)
# ax7.set_facecolor(background_color)
# ax8.set_facecolor(background_color)

# # Title of the plot
# ax0.spines["bottom"].set_visible(False)
# ax0.spines["left"].set_visible(False)
# ax0.spines["top"].set_visible(False)
# ax0.spines["right"].set_visible(False)
# ax0.tick_params(left=False, bottom=False)
# ax0.set_xticklabels([])
# ax0.set_yticklabels([])
# ax0.text(0.5, 0.5,
#          'Count plot for various\n categorical features\n_________________',
#          horizontalalignment='center',
#          verticalalignment='center',
#          fontsize=18, fontweight='bold',
#          fontfamily='serif',
#          color="#000000")

# # Function to create count plot using bar
# def create_count_plot(ax, data, x, palette):
#     ax.bar(data[x].value_counts().index, data[x].value_counts().values, color=palette)
#     ax.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
#     ax.set_xlabel("")
#     ax.set_ylabel("")

# # Sex count
# ax1.text(0.3, 165, 'Sex', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax1, df, 'sex', color_palette)
# ax1.set_xticks([0, 1])
# ax1.set_xticklabels(["Female(0)", "Male(1)"])

# # Exng count
# ax2.text(0.3, 160, 'Exng', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax2, df, 'exng', color_palette)
# ax2.set_xticks([0, 1])
# ax2.set_xticklabels(["No(0)","Yes(1)"])

# # Caa count
# ax3.text(1.5, 120, 'Caa', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax3, df, 'caa', color_palette)
# ax3.set_xticks([0, 1,2,3,4])


# # Cp count
# ax4.text(1.5, 120, 'Cp', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax4, df, 'cp', color_palette)
# ax4.set_xticks([0,1,2,3])
# ax4.set_xticklabels(["Typical angina(1)","Atypical angina(2)","nopain(3)","asymptomatic(4)"], rotation=15)

# # Fbs count
# ax5.text(0.5, 200, 'Fbs', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax5, df, 'fbs', color_palette)
# ax5.set_xticks([0, 1])
# ax5.set_xticklabels(["False(0)","True(1)"])

# # Restecg count
# ax6.text(0.75, 120, 'Restecg', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax6, df, 'restecg', color_palette)
# ax6.set_xticks([0, 1,2])
# ax6.set_xticklabels(["normal(0)","ST-T abnormality (1)","LV hypertrophy(2)"], rotation=15)

# # Slp count
# ax7.text(0.85, 120, 'Slp', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax7, df, 'slp', color_palette)
# ax7.set_xticks([0, 1,2])

# # Thall count
# ax8.text(1.2, 120, 'Thall', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# create_count_plot(ax8, df, 'thall', color_palette)
# ax8.set_xticks([0, 1, 2, 3])

# # Remove spines
# for s in ["top", "right", "left"]:
#     ax1.spines[s].set_visible(False)
#     ax2.spines[s].set_visible(False)
#     ax3.spines[s].set_visible(False)
#     ax4.spines[s].set_visible(False)
#     ax5.spines[s].set_visible(False)
#     ax6.spines[s].set_visible(False)
#     ax7.spines[s].set_visible(False)
#     ax8.spines[s].set_visible(False)


# st.pyplot(fig)

# html_5 = """
# <div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
# <center><h3>Count of the target</h3></center>
# </div>
# """
# st.markdown(html_5, unsafe_allow_html=True)
# st.markdown("")



# fig = plt.figure(figsize=(18, 7))
# gs = fig.add_gridspec(1, 2)
# gs.update(wspace=0.3, hspace=0.15)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[0, 1])

# background_color = "#ffe6e6"
# color_palette = ["#800000", "#8000ff", "#6aac90", "#5833ff", "#da8829"]
# fig.patch.set_facecolor(background_color)
# ax0.set_facecolor(background_color)
# ax1.set_facecolor(background_color)

# # Title of the plot
# ax0.text(0.5, 0.5, "Count of the target\n___________",
#          horizontalalignment='center',
#          verticalalignment='center',
#          fontsize=18,
#          fontweight='bold',
#          fontfamily='serif',
#          color='#000000')

# ax0.set_xticklabels([])
# ax0.set_yticklabels([])
# ax0.tick_params(left=False, bottom=False)

# # Target Count
# ax1.text(0.35, 130, "Output", fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
# ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
# ax1.bar(df['output'].value_counts().index, df['output'].value_counts().values, color=color_palette)
# ax1.set_xlabel("")
# ax1.set_ylabel("")
# ax1.set_xticks([0, 1])
# ax1.set_xticklabels(["Low chances of attack(0)", "High chances of attack(1)"])

# # Remove spines
# for s in ["top", "left", "right"]:
#     ax0.spines[s].set_visible(False)
#     ax1.spines[s].set_visible(False)
# st.pyplot(fig)


### Analysis ###

html_6 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h3>Example data table</h3></center>
</div>
"""
st.markdown(html_6, unsafe_allow_html=True)
st.markdown("")
st.write(df.head(10))




html_7 = """
<div style="background-color:#0E1117;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;margin-top:20px;">
<center><h3>Prediction</h3></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")


s1 = st.number_input("อายุของผู้ป่วย")
s2 = st.selectbox("โรคโรหิตจาง (0 : ไม่เป็น | 1 : เป็น)",[0,1])
s3 = st.number_input("ระดับเอนไซม์CPKในเลือด(mcg/L)")
s4 = st.selectbox("โรคเบาหวาน (0 : ไม่เป็น | 1 : เป็น)",[0,1])
s5 = st.number_input("อัตราส่วนร้อยละของเลือดที่สูบออกจากหัวใจในแต่ละครั้ง(ร้อยละ)")
s6 = st.selectbox("โรคความดันโลหิตสูง (0 : ไม่เป็น | 1 : เป็น)",[0,1])
s7 = st.number_input("จำนวนเกร็ดเลือดในเลือด (หน่วย กิโลเกร็ดเลือด/มิลลิลิตร)")
s8 = st.number_input("ปริมาณซีรั่มครีเอทินีนในเลือด (มิลลิกรัม/เดซิลลิตร)")
s9 = st.number_input("ปริมาณซีรั่มโซเดียมในเลือด (มิลลิเอควิเทลนต์/ลิตร)")
s10 = st.selectbox("เพศ (0 : ผู้หญิง | 1 : ผู้ชาย)",[0,1])
s11 = st.selectbox("การสูบบุหรี่ (0 : ไม่ได้สูบ | 1 : สูบ)",[0,1])
s12 = st.number_input("ระยะเวลาในการติดตามผู้ป่วย")

if st.button("ทำนายผล"):
   
   X=df.drop(["DEATH_EVENT"],axis=1)
   y=df["DEATH_EVENT"]

   X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=7)
   ds_model = DecisionTreeClassifier()
   ds_model.fit(X, y)
   x_input = np.array([[s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]])

   out = ds_model.predict(x_input)

   if out[0]== 0 :
          
      html_8 = """
      <div style="background-color:#0E1117;padding:20px;border: 3px solid #ffffff;">
      <center><h3 style="border-bottom: 3px solid #ffffff;">ไม่มีความเสี่ยงในการเสียชีวิต</h3></center>

      </div>
      """
      st.markdown(html_8, unsafe_allow_html=True)
      st.markdown("")

   elif out[0]==1:
          
          
      html_9 = """
      <div style="background-color:#0E1117;padding:20px;border: 3px solid #ffffff;">
      <center><h3 style="border-bottom: 3px solid #ffffff;">มีความเสี่ยงในการเสียชีวิต</h3></center>
      <left><h6 style="text-indent: 30px;line-height: 1.5;padding-top:15px;">ภาวะหัวใจล้มเหลวเป็นภาวะแทรกซ้อนที่พบบ่อยของโรคหัวใจและหลอดเลือด จากข้อมูลของผู้ป่วยที่เสียชีวิตพบว่าค่าเฉลี่ยของข้อมูลต่างๆมีดังนี้ สามารถใช้ข้อมูลเหล่านี้เพื่อนำไปอ้างอิงเพื่อลดความเสี่ยงให้กับผู้ป่วย</h6></left>
      <ul>
         <li>จากข้อมูล อายุเฉลี่ยของผู้ป่วยที่เสียชีวิตจากภาวะหัวใจล้มเหลวคือ 68 ปี</li>
         <li>จากข้อมูล ค่าเฉลี่ยของเอนไซม์CPKในเลือดของผู้ป่วยที่เสียชีวิตคือ 670mcg/L สำหรับผู้ที่มีความเสี่ยงชีวิตมากแนะนำให้ลดปริมาณเอนไซม์CPKลง</li>
         <li>จากข้อมูล ร้อยละของเลือดที่สูบออกจากหัวใจในแต่ละครั้ง(ejection fraction)หรือเศษอีเจ็คชันโดยปกติแล้วในคนปกติจะมีค่าอยู่ที่ร้อยละ60 ส่วนผู้ป่วยที่เสียชีวิตอยู่ที่ร้อยละ30 การรักษาที่อาจช่วยเพิ่มเศษอีเจ็คชัน ได้แก่ ยา ยาบางชนิด เช่น ยาขยายหลอดเลือดหัวใจ การผ่าตัด การผ่าตัดบางประเภท เช่น การผ่าตัดบายพาสหัวใจหรือการผ่าตัดรักษาโรคลิ้นหัวใจผิดปกติ สามารถช่วยปรับปรุงการทำงานของหัวใจและเพิ่มเศษอีเจ็คชัน การเปลี่ยนแปลงวิถีชีวิต การเปลี่ยนแปลงวิถีชีวิต เช่น การรับประทานอาหารเพื่อสุขภาพ การลดน้ำหนัก การเลิกสูบบุหรี่ และการออกกำลังกายอย่างสม่ำเสมอ สามารถช่วยปรับปรุงการทำงานของหัวใจและเพิ่มเศษอีเจ็คชัน</li>
         <li>จากข้อมูล ค่าเฉลี่ยจำนวนเกร็ดเลือดในเลือด (หน่วย กิโลเกร็ดเลือด/มิลลิลิตร)ของผู้ป่วยที่เสียชีวิตคือ 256381 กิโลเกร็ด/มิลลิลิตร</li>
         <li>จากข้อมูล ค่าเฉลี่ยปริมาณซีรั่มครีเอทินีนในเลือด (มิลลิกรัม/เดซิลลิตร)ของผู้ป่วยที่เสียชีวิตคือ 1.83 มิลลิกรัม/เดซิลลิตร </li>
         <li>จากข้อมูล ค่าเฉลี่ยปริมาณซีรั่มโซเดียมในเลือด (มิลลิเอควิเทลนต์/ลิตร)ของผู้ป่วยที่เสียชีวิตคือ 135 มิลลิเอควิเทลนต์/ลิตร</li>
      </ul>
      <left><h6 style="text-indent: 30px;line-height: 1.5;padding-top:15px;">การปฏิบัติตนตามแนวทางดังกล่าวอาจจะช่วยลดโอกาสที่จะเกิดภาวะหัวใจล้มเหลวจนทำให้เกิดการเสียชีวิตได้. อย่าลืมปรึกษาแพทย์หากคุณมีปัญหาสุขภาพหรือต้องการคำแนะนำเพิ่มเติมเกี่ยวกับการรักษาและป้องกันโรคหัวใจและหลอดเลือด</h6></left>
      </div>
      """
      st.markdown(html_9, unsafe_allow_html=True)
      st.markdown("")
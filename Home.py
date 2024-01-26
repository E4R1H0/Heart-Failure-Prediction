import streamlit as st


col1, col2, col3 = st.columns([1.5, 6, 1])

with col1:
    st.write("") 

with col2:
    st.image("./pic/dataset-cover.jpg")

with col3:
    st.write("")


html_1 = """
<div style="background-color:#0E1117;margin-top:40px;padding:5px;border-radius:5px;border-bottom: 3px solid #ffffff;border-top: 3px solid #ffffff;">
<center><h4 style ="filter: drop-shadow(0px 0px 5px #89c5fb);">ระบบวิเคราะห์สุขภาพการนอนหลับและไลฟ์สไตล์โดยใช้เทคนิคเหมืองข้อมูล</h4><h5  style ="filter: drop-shadow(0px 0px 5px #89c5fb);">The system analyzes health, sleep and lifestyle using data mining techniques</h5></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

html_2 = """
<div style="background-color:#0E1117;padding:15px;">
<center></center><left><h6 style="text-indent: 30px;line-height: 1.5;">การรักษาสุขภาพการนอนมีความสำคัญสำหรับความเป็นอยู่ของบุคคลทั้งทางร่างกายและจิตใจทุกวัย การนอนไม่เพียงเติมพลังให้ร่างกายและสมองเท่านั้น แต่ยังมีผลที่สำคัญต่อระบบฮอร์โมน การฟื้นตัวของร่างกาย และส่งเสริมสุขภาพจิตใจด้วย การนอนไม่เพียงพอสามารถทำให้ความต้านทานลดลง, มีปัญหาการคิด, และการเรียนรู้มีผลกระทบ นอนน้อยยังเสี่ยงต่อโรคต่างๆ เช่น โรคอ้วน, โรคหัวใจ, วิตกกังวล, หรือโรคจิตเร้าอารมณ์, และภาวะซึมเศร้า การวิเคราะห์ข้อมูลการนอนเป็นทางการเพื่อให้คนทั่วไปทราบถึงปัญหาของตนเอง การใช้ 3 เทคนิคในการวิเคราะห์ได้แก่ เพื่อนบ้านใกล้สุด (K-nearest Neighbors), ต้นไม้ตัดสินใจ (Decision Tree), และ นาอีฟเบย์ (Naive Bayes) เป็นเครื่องมือที่มีประสิทธิภาพในการนำข้อมูลมาวิเคราะห์ ผลการวิเคราะห์นี้จะช่วยให้บุคคลที่สนใจสามารถเข้าใจถึงสถานะของการนอนของตนเอง และสามารถตัดสินใจดำเนินการรักษาหรือปรับเปลี่ยนพฤติกรรมการนอนในทางที่เหมาะสม เพื่อรักษาสุขภาพทั้งร่างกายและจิตใจให้มีคุณภาพดีที่สุด ผลการวิเคราะห์พบว่าเทคนิค Decision Tree มีความแม่นยำสูงที่สุดที่ระดับ 0.89, ตามด้วย Naive Bayes ที่มีความแม่นยำที่ 0.88 และ K-nearest Neighbors ที่มีความแม่นยำที่ 0.87 จากผลลัพธ์นี้, ผู้วิจัยจึงเลือกใช้ Decision Tree เป็นเทคนิคในการพัฒนาระบบทำนายสถานะการนอนของผู้ใช้งานในเว็บไซต์ ทั้งนี้, ผู้ใช้งานสามารถทดสอบระบบเพื่อตรวจสอบสถานะการนอนของตนเองได้</h6></left>
<center></center><left><h6 style="text-indent: 30px;line-height: 1.5;">Maintaining good sleep health is crucial for overall well-being, both physically and mentally, at all ages. Sleep not only replenishes energy for the body and brain but also plays a significant role in hormone regulation, body recovery, and mental health promotion. Insufficient sleep can lead to weakened immunity, cognitive problems, and impaired learning. Additionally, inadequate sleep is associated with various health risks, such as obesity, heart disease, anxiety disorders, depression, and even conditions like sleep disorders.

Analyzing sleep data formally can provide valuable insights into an individual's sleep issues. Using three effective techniques—K-nearest Neighbors, Decision Tree, and Naive Bayes—facilitates the analysis process. These tools efficiently process data to help individuals understand their sleep patterns better. Sleep deprivation can result in reduced resistance, cognitive difficulties, and increased susceptibility to various diseases.

The results of the analysis indicate that Decision Tree has the highest accuracy level at 0.89, followed by Naive Bayes at 0.88 and K-nearest Neighbors at 0.87. Based on these findings, researchers have chosen to implement the Decision Tree technique to develop a predictive system for users' sleep status on a website. Users can utilize this system to assess their own sleep status and make informed decisions to improve their sleep habits, ultimately enhancing both physical and mental well-being.</h6></left>
</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
st.markdown("")

html_3 = """
<div style="background-color:#0E1117;border-top: 3px solid #ffffff;">
<center><h4 style="text-indent: 50%;">Wiriya Hemmala 64/44 007</h4></center>
</div>
"""
st.markdown(html_3, unsafe_allow_html=True)
st.markdown("")
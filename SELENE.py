import streamlit as st
from utilities.icon import page_icon
st.set_page_config(
    page_title="SELENE",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
)

st.header("SELĒNE")
st.subheader("Problem Statement")
st.write("Many people suffer with rheumatoid arthritis which is a life long incurable disease, and along with this they suffer with mental health issues that plague their lives and these people can’t find a holistic support for the problems that they face in their lives.")

st.markdown("---")

st.subheader("Team Members")
st.write("Sarika: Founder, CEO of SELENE")
st.write("Roshni: UI/UX Designer Lead and Researcher")
st.write("Richard: Co-Founder, CTO of SELENE")

st.markdown("---")

st.subheader("Prevelance")
st.write("Rheumatoid arthritis (RA) has a prevalence rate of approximately 0.7% in the adult Indian population, which is higher than the global prevalence of 0.46%. A study near Delhi projected a prevalence of 0.75%, translating to around seven million RA patients in India. ")

st.markdown("---")

st.header("Additional Information")
st.write("Contact Us: 8885581219")

st.markdown("Thank you for visiting our Page!")

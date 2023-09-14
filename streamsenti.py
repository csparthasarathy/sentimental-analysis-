import streamlit as st
import joblib

st.sidebar.title("Mental Health Classifier")
st.sidebar.write("Enter your text and see if it's stressful or not!")

st.title("Mental Health Classifier")

user_input = st.text_area("Enter your text:")

if st.button("Analyse"):
    if user_input:
    
        classifier = joblib.load('senti.pkl')
        tfidf = joblib.load('data_vec.pkl')
    
        user_input_tfidf = tfidf.transform([user_input])

        y_new = classifier.predict(user_input_tfidf)[0]

        if y_new == 1:
            st.write("Your text appears to convey a sense of tranquility and ease, indicating that you're not currently feeling stressed.Keep up the positive vibes :) ")
        else:
            st.write("**Your text suggests that you may be experiencing some stress**. It's essential to prioritize self-care. Take regular breaks, engage in relaxation techniques, and don't hesitate to reach out to friends or professionals for support. Additionally, consider exploring the resources available on our homepage, including music, reading materials, and yoga therapy. These can be helpful in managing stress and promoting well-being.")

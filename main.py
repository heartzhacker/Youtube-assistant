import streamlit as st
import langchain_helper as lch
import textwrap 

st.title("Youtube assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label = "What is the link to youtube video",
            max_chars=100
        )
        query = st.sidebar.text_area(
            label="Ask me anything based on the video url",
            max_chars= 1000,
            key = "query"
        )
        openai_api_key = st.sidebar.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        

        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lch.create_vector_db_yt_yrl(youtube_url)
    response = lch.get_response(db,query,openai_api_key)
    st.subheader("Answer: ")
    st.text(textwrap.fill(response, width = 60))


from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Langchain Udemy Course - Doc Helper Bot")

prompt = st.text_input("Prompt",placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

def create_sources_response(source_urls:set[str]) -> str:
    if not source_urls:
        return ""
    source_list = list(source_urls)
    source_list.sort()
    source_string = "sources:\n"
    for i,source in enumerate(source_list):
        source_string += f"{i+1}. {source}\n"
    return source_string
    

if prompt:
    with st.spinner("Generation response..."):
        generate_response = run_llm(query=prompt)
        sources = set([doc.metadata["source"] for doc in generate_response["source_documents"]])

        formatted_response = f"{generate_response['result']} \n\n {create_sources_response(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response,user_query in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
        message(user_query,is_user=True)
        message(formatted_response)

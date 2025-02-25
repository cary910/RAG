import streamlit as st
from utils_P4复现01 import qa_agent
from langchain.memory import ConversationBufferMemory

st.title("智能PDF问答工具")
with st.sidebar:
    api_key = st.text_input("请输入您的OpenAI API密钥：",type="password")
    st.markdown("[获取API密钥地址](https://openai.com/account/api-keys")

if "memory" not in st.session_state:
    st.session_state["memory"]=ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
uploaded_file=st.file_uploader("请上传您的PDF文档",type="pdf")
question = st.text_input("请对PDF提出问题",disabled = not uploaded_file)

if uploaded_file and question and not api_key:
    st.info("请输入您的API密钥")

if uploaded_file and question and api_key:
    with st.spinner("AI正在思考，请稍候..."):
        response = qa_agent(api_key,st.session_state["memory"],uploaded_file,question)
        st.write("### 答案")
        st.write(response["answer"])
        st.session_state["chat_history"] = response["chat_history"]

    if "chat_history" in st.session_state:
        with st.expander("历史消息"):
            for i in range(0,len(st.session_state["chat_history"]),2):
                human_message=st.session_state["chat_history"][i]
                ai_message=st.session_state["chat_history"][i+1]
                st.write(human_message.content)
                st.write(ai_message.content)
                if i < len(st.session_state["chat_history"])-2:
                    st.divider()
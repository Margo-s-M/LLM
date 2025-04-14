from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ( SystemMessage, HumanMessage, AIMessage,trim_messages)
import streamlit as st



llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=st.secrets.get('GEMINI_API_KEY'),
)
#print(st.secrets.get('GEMINI_API_KEY'))

st.title("ITStep Chat-bot")
st.markdown('Привіт Я чат бот для спілкування. Модель gemini-2.0-flash')
user_input = st.chat_input("Привіт давай по спілкуємося ")
# st.markdown(user_input)
# print(user_input)
#додаємо історію до сесії
#якщо початок сесії
if 'messages' not in st.session_state:
 st.session_state.message = [
    SystemMessage(content="""Ти ввічливий чат бот.Відповідай коротко та чітко""")
]
if user_input is not None:
    human_message = HumanMessage(content=user_input)
    st.session_state.message.append(human_message)
    #ВИКЛИК МОДЕЛІ
    response = llm.invoke((st.session_state.message))
    print(response)

    #додати відповідь модель
    st.session_state.message.append(response)

    #відображення історії спілкування
for message  in st.session_state.message:
    #перевіряємо хто писав повідомлення
    if isinstance(message,HumanMessage):  #перевірка на тип данних
        role = 'human'
    elif isinstance(message, AIMessage):
        role = 'ai'
    else:
        continue

    with st.chat_message(role):
        st.markdown(message.content)

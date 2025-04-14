# Напишіть додаток, який симулює спілкування з певною
# відомою людиною.
# З ким саме спілкуватись вводить користувач через
# st.text_input()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ( SystemMessage, HumanMessage, AIMessage,trim_messages)
import streamlit as st



llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=st.secrets.get('GEMINI_API_KEY'),
)

st.title("Chat-bot відома людина")
st.markdown('Введіть імя відомої особи з якою ви хочете поспілкувватися')
person =st.text_input('Вкажіть імя ')
print(person)
user_input = st.chat_input("Привіт давай по спілкуємося ")
if 'message' not in st.session_state:
 st.session_state.message = []

if st.button('Старт чат'):
 st.session_state.message = [
    SystemMessage(content=f"""Ти імітуєш відому людину {person}.Відповідай лаконічно.""")
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
for message in st.session_state.message:
    #перевіряємо хто писав повідомлення
    if isinstance(message,HumanMessage):  #перевірка на тип данних
        role = 'human'
    elif isinstance(message, AIMessage):
        role = 'ai'
    else:
        continue

    with st.chat_message(role):
        st.markdown(message.content)
print(len(st.session_state.message))
import dotenv
import os
import warnings

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate


warnings.filterwarnings('ignore') # ігнорувати warnings
dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    temperature = 0.3,
)


"Завдання 1 Напишіть промпт для створення плану навчального курсу з певної теми "
"для цільової айдиторії(початківці, професіонали, діти, тощо)."
"Вхідні параметри: тема, опис цільової аудиторії "
"Реалізуйте двома способами: "
" Zero-shot "
" Few-shot "

prompt = PromptTemplate. from_template("""
[INST]
Ти викладач-методист з англійської мови.
Створи план навчального курсу.

Тема плану: "Business English".
Цільова аудиторія: професіонали,

[/INST]
""")
result= prompt.format()

out_put = llm.invoke(result)
print(out_put)


#Few-shot промпт,
# Ти викладач-методист на курсах з вивчення англійської мови.
# [INST]
# Створи план навчального курсу.
# Тема: "Technical English for IT Specialists"
# Аудиторія: Програмісти, ІТ-фахівці.
# Приклад як ти маєш це виконати:
# Module 1: Introduction to IT English
# Goal: Master basic vocabulary for developers.
# Topics: Technical terms, IT company structure.
# Practice: Writing emails in English.
# Resources: Coursera – "English for IT".
# [/INST]

"""Створіть найпростіший чат бот.
Напишіть моделі якого персонажа вона повинна вдавати(відомий актор, персонаж кіно\книги, тощо).
 Реалізуйте двома способами:
1. Модель отримує інструкцію в якому стилі відповідати та нове повідомлення.
2. Модель отримує інструкцію та історію попередніх повідомлень як від користувача,
так і її власні відповіді у форматі Instruction: ….
Human: massage1
AI: message2 Human:
 massage3 AI: message4
 Human: massage5 AI:  """



import dotenv
import os

from  langchain_huggingface import HuggingFaceEndpoint


dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    temperature= 0.8, #креативність
    max_new_tokens= 120  #довжина відповіді від ші
)




def chat():
    instruction = (
        """Ти -Вінні Пух.Ти ти персонаж мутьтфільму.Твої відповіді кумедні.Ти дуже добри  та  розумний, 
    любиш всі окрім бджіл.Ти маєш гарну підтримку від друзів Кролика,Тигрюлі, Хрюника та віслюка Вушастика.
    В тебе є накращий друг Крістофер Робін.
     """
    )
    history = [] #збереження діалогу

    while True:
        try:
            user_message = input("Привіт , давай поговоримо, коли захочеш завешити діалог просто напиши -прощавай-: ")
            if user_message.lower() == "прощавай":
                print("Чат завершено.")
                break

            history.append(f"Human: {user_message}")  #додаємо до історії
            history_text = "\n".join(history) #текст промтру на фоні історії
            prompt = f"[INST]{instruction}\n{history_text}\nAI: [/INST]" #промт для моделі

            response = llm.invoke(prompt)
            ai_response = response.strip()
            print(f"Вінні Пух: {ai_response}")

            history.append(f"AI:{ai_response}") # +відповідьбота до історії
        except Exception as e:
            print(f"Сталася помилка:{e}")

if __name__== "__main__":
    chat()


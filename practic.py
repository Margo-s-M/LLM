import dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint

dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    temperature= 0.2,
    max_new_tokens= 25

)


# Завдання 2
# Прочитайте файл data\lesson9\rules.txt з правилами
# користування атракціону. Напишіть програму яка отримує від
# користувачі питання та дає відповідь на нього виходячи з
# текстового файлу.
# Для цього об’єднайте правила користування з питанням
# користувача.
# Користувач задає питання поки не введе порожній рядок.
# Змініть файл rules.txt, щоб переконатись що модель дійсно
# його читає

with open("rules.txt", "r", encoding="utf-8") as file:
    result = file.read()
    #print(result)

responce= llm.invoke(f"""[INST]Ти інструктор з правил користування атракціоном.
Давай короткі відповіді без уточнень
Правила: {result}
Якщо мені 10 років, чи можу я піти на атракціон?[/INST]""")

print(responce)
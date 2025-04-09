# Завдання 3
# Напишіть чат бота, який допомагає у вивченні англійської мови з наступним функціоналом:
#  якщо користувач просить перекласти слово або фразу то
# дається переклад слова та приклад використання в реченні
#  якщо користувач просить перекласти речення, то
# дається переклад самого речення, а також пояснення граматики,
# наприклад структура there is\are,
# питання в різних часових формах, тощо.
# Приклади реалізуйте як HumanMessage та AIMessage
# Завдання4
# Модифікуйте попереднє завдання таким чином, щоб в SystemMessage користувачем.
# передавався список вивчених слів
# Для цього напишіть окрему модель яка буде діставати з відповіді(AIMessage)
# усі англійські слова(вважаємо що користувач знає лише ті слова, про які йому сказала модель).
# Список вивчених слів треба зберігати в json файлі та відвантажувати при запуску програми.
# Змініть функціонал таким чином:  якщо користувач просить перекласти слово або фразу то
# дається переклад слова та приклад використання в реченні з вивченими словами
#  якщо користувач просить перекласти речення, то додатково пояснюється значення невідомих слів

import dotenv
import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
import json

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=os.getenv('GEMINI_API_KEY')
)


trimmer = trim_messages(
    strategy= "last",
    token_counter= len,
    max_tokens=5,
    start_on= "human",
    end_on="human",
    include_system = True
)
messages = [
    SystemMessage(content='Ти чат бот, який допомагає вивчати англійську мову.'
                          'Давай короткі та чіткі відповіді.За проханням користувача пояснюй граматику.'
                          'Надай приклади використання нових слів в реченнях.'
)]


#Завантаження списку вивчених слів з JSON
def load_learned_words(path="learned_words.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Збереження списку вивчених слів у JSON
def save_learned_words(words, path="learned_words.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(set(words))), f, ensure_ascii=False, indent=2)

# Витяг англійських слів з тексту
def extract_english_words(text):
    return re.findall(r"\b[a-zA-Z]{2,}\b", text)

# Генерація повідомлень з урахуванням логіки завдання
def get_prompt_with_system(user_input, learned_words):
    system = SystemMessage(
        content=f"Користувач знає наступні англійські слова: {', '.join(sorted(set(learned_words)))}"
    )

    # Якщо це речення (закінчується на крапку чи питання)
    if user_input.strip().endswith(('.', '?')):
        human = HumanMessage(content=f"""
        Переклади речення з української: "{user_input}"
        Додай пояснення граматики (наприклад, there is/are, запитання в часах )
        Також поясни значення нових слів, які ще не входять до списку вивчених.
        """)
    else:
        human = HumanMessage(content=f"""
        Переклади слово або фразу з української: "{user_input}".
        Склади приклад речення, використовуючи тільки вивчені слова: {', '.join(learned_words)}.
        """)

    return [system, human]

# Головна логіка
def main():
    learned_words = load_learned_words()
    messages = []

    chain = trimmer | llm

    while True:
        user_input = input("Let's learn English!: ")

        if user_input.strip() == '':
            break

        # Генерація повідомлень
        prompt_messages = get_prompt_with_system(user_input, learned_words)

        # Додаємо system + human до історії
        messages.extend(prompt_messages)

        # Отримуємо відповідь
        response = chain.invoke(messages)

        # Додаємо AIMessage до історії
        messages.append(response)

        # Вивід
        print(f"\nAI: {response.content}\n")

        # Оновлення списку вивчених слів
        new_words = extract_english_words(response.content)
        learned_words.extend([word.lower() for word in new_words])
        learned_words = list(set(learned_words))  # унікальність

        # Зберігаємо оновлений список
        save_learned_words(learned_words)

        # Вивід історії (опційно)
        print("Історія повідомлень:")
        for msg in messages:
            print(f"{msg.type.upper()}: {msg.content.strip()[:100]}...")

        print("Вивчені слова:", ', '.join(sorted(learned_words)))
        print("-" * 40)

# Запуск
if __name__ == "__main__":
    main()
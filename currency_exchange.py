# Напишіть модель
# яка конвертує одну валюту в іншу за нинішнім курсом.
# Для цього напишіть функції,
# яка отримує номінал та курс і робить конвертацію.
# Реалізуйте 2 ланцюга:
#  перший отримує назви валют та шукає курс в інтернеті
#  другий отримує номінал та курс і застосовує функцію ковертації

import os
import dotenv
import re
import requests
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# --- 1. Завантаження змінних середовища ---
dotenv.load_dotenv()

# --- 2. LLM модель ---
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=os.getenv('GEMINI_API_KEY'),
)

# --- 3. Функція автоматичної конвертації ---
def auto_currency_convert_api(input_str: str) -> str:
    """
    Приймає рядок типу 'Конвертуй 100 USD в EUR' і повертає результат.
    """
    match = re.search(r'(\d+(?:\.\d+)?)\s*([A-Z]{3})\s*в\s*([A-Z]{3})', input_str.upper())
    if not match:
        return "Формат запиту має бути: 'Конвертуй 100 USD в EUR'"

    amount = float(match.group(1))
    from_currency = match.group(2)
    to_currency = match.group(3)

    url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}&amount={amount}"
    response = requests.get(url)

    if response.status_code != 200:
        return "Помилка при отриманні даних з API."

    data = response.json()
    result = round(data['result'], 2)
    rate = round(data['info']['rate'], 4)

    return f"{amount} {from_currency} = {result} {to_currency} за курсом {rate}"

# --- 4. Створення Tool ---
currency_conversion_tool = Tool(
    name="CurrencyConverter",
    func=auto_currency_convert_api,
    description="Автоматично конвертує валюту. Наприклад: 'Конвертуй 100 USD в EUR'"
)

# --- 5. Додаємо Tool до агента ---
tools = [currency_conversion_tool]
agent = create_react_agent(model=llm, tools=tools)

# --- 6. Стартове повідомлення та сесія ---
data_input = {
    'messages': [
        SystemMessage(content='Ти помічник у фінансових питаннях. Твоя задача — конвертувати валюту за запитом користувача. '
                              'Використовуй інструмент CurrencyConverter, коли користувач просить "Конвертуй 100 USD в EUR" або подібне.'),
    ]
}

# --- 7. Цикл взаємодії ---
while True:
    user_input = input("YOU: ")
    if user_input == "":
        break

    data_input['messages'].append(HumanMessage(content=user_input))
    response = agent.invoke(data_input)
    data_input = response

    print("🤖:", data_input['messages'][-1].content)

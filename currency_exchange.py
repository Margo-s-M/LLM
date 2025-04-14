# Напишіть модель
# яка конвертує одну валюту в іншу за нинішнім курсом.
# Для цього напишіть функції,
# яка отримує номінал та курс і робить конвертацію.
# Реалізуйте 2 ланцюга:
#  перший отримує назви валют та шукає курс в інтернеті
#  другий отримує номінал та курс і застосовує функцію ковертації

import os
import dotenv
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage


dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=os.getenv('GEMINI_API_KEY'),
)

# === 1. Конвертація валют ===
def convert_currency(nominal: float, rate: float) -> float:
    """
    Конвертує валюту за курсом.

    :param nominal: сума в початковій валюті
    :param rate: курс до цільової валюти
    :return: конвертована сума
    """
    return round(nominal * rate, 2)


# === 2. Функція-помічник, яка робить конвертацію та вивід ===
def currency_conversion_tool(input_str: str) -> str:
    """
    Очікує на вхід рядок у форматі: "100, 38.5"
    Де 100 — це сума, 38.5 — курс.

    :return: Результат конвертації.
    """
    try:
        nominal_str, rate_str = input_str.split(',')
        nominal = float(nominal_str.strip())
        rate = float(rate_str.strip())
        result = convert_currency(nominal, rate)
        return f"{nominal} * {rate} = {result}"
    except Exception as e:
        return f"Помилка у вводі: {e}. Очікуваний формат: 'номінал, курс'"


# === 3. Tool для пошуку ===
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Використовуй для пошуку курсу валют в інтернеті.",
    ),
    Tool(
        name="CurrencyConverter",
        func=currency_conversion_tool,
        description="Конвертує валюту. Введи 'номінал, курс' напр. '100, 38.5'",
    )
]

# === 4. Створення агента ===
agent = create_react_agent(model=llm, tools=tools)

data_input = {
    'messages': [
        SystemMessage(content='Ти помічник у фінансових питаннях. '
                              'Ти можеш шукати курс валют та робити конвертацію. '
                              'Щоб конвертувати, використай інструмент CurrencyConverter. '
                              'Щоб знайти курс, використай інструмент Search. '
                              'Спочатку дізнайся курс, потім конвертуй валюту.'),
    ]
}

# === 5. Запуск циклу спілкування ===
while True:
    user_input = input('YOU: ')
    if user_input == '':
        break

    data_input['messages'].append(HumanMessage(content=user_input))
    response = agent.invoke(data_input)
    data_input = response
    print(f": {data_input['messages'][-1].content}")

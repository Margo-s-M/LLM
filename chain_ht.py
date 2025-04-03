"""Напишіть модель для генерації персонального плану тренувань
 з двох ланцюгів:
  Перший ланцюг отримує мету тренування(схуднення, набір м’язів, тощо)
 та повертає список вправ
  Другий ланцюг отримує список вправ,
 рівень підготовки користувача(низький, середній, професіонал)
 та кількість часу на тиждень(в годинах) і
 повертає план тренувань """

import dotenv
import warnings

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


warnings.filterwarnings('ignore')  # ігнорувати warnings
dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    temperature=0.2,
)

schemas1 = [
    ResponseSchema(name='список вправ', description='список вправ згідно мети тренування'),#Перший ланцюг отримує мету тренування(схуднення, набір м’язів, тощо)
    ]
parser1 = StructuredOutputParser.from_response_schemas(schemas1)
format_instructions = parser1.get_format_instructions()

prompt1 = PromptTemplate.from_template("""
[INST]
Ти фітнес тренер.
Твоє завдання -скласти список фізичних вправ.
Взалежності від вибору користувача ти складаєш план за метою тренування.
Метою може бути: схуднення, набір м’язівб,покращення здоровя, тощо.
Надай структурований список вправ.
Відповідай Українською мовою.

[/INST]

[INST]
Мета тренування:{мета тренування}
Формат відповіді:{format_instructions}
Відповідь:

[/INST]

""",
    partial_variables={'format_instructions': format_instructions})

chain1 = prompt1 | llm | parser1


schemas2 = [
    ResponseSchema(name='план тренування', description='Детальний план тренування з урахування всіх факторів'), #Другий отримує отримує список вправ,
 # рівень підготовки користувача(низький, середній, професіонал)
 # та кількість часу на тиждень(в годинах) і
 # повертає план тренувань

    ]
parser2 = StructuredOutputParser.from_response_schemas(schemas2)
format_instructions = parser2.get_format_instructions()

prompt2 = PromptTemplate.from_template("""
[INST]
Ти фітнес тренер.
На основі наданого списку вправ, рівня підготовки користувача 
та кількості годин тренування , творити персоналізований план тренувань.
Рівень підготовки запитай у користувача
Відповідь має бути структурованою та чіткою.
Використовуй Українську мову для відповіді.

[/INST]

[INST]
Список вправ:{список вправ}
Формат відповіді:{format_instructions}
Відповідь:

[/INST]

""",
    partial_variables={'format_instructions': format_instructions})

chain2 = prompt2| llm | parser2

user_input = input(" Для того щоб я створив ваш індивідуальний план тренування вкажіть ваші побажання :")

data = {'мета тренування': user_input }

response1 = chain1.invoke(data)
print(response1)

response2 = chain2.invoke(response1)
print(response2)

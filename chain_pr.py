# Напишіть модель для рекомендації книг з двох ланцюгів:
#  Перший ланцюг отримує назву книги та визначає її
# жанр
#  Другий отримує назву книги, жанр та повертає список
# схожих книг(того ж самого жанру та іншого)


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


schemas = [
    ResponseSchema(name='жанр', description='Жанр в якому написана книга'),#Перший ланцюг отримує назву книги та визначає її жанр
    ]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""
[INST]
Ти помічник редактора.
Твоя задача полягає в тому аби визначати жанр книги за назвою.
Відповідь має бути коротка.
Використовуй не більше одного речення.
Використовуй Українську мову для відповіді.

[/INST]

[INST]
Нава книги:{назва}
Формат відповіді:{format_instructions}
Відповідь:

[/INST]

""",
    partial_variables={'format_instructions': format_instructions})

chain1 = prompt | llm | parser


schemas = [
    ResponseSchema(name='список', description='список схожих книг'), #Другий отримує назву книги, жанр та повертає список
# схожих книг(того ж самого жанру та іншого)
    ]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""
[INST]
Ти помічник редактора.
Твоя задача полягає в тому аби за вказаним жанром порекомендувати 5 книг такого ж жанру.
Відповідь має бути короткою.
Використовуй лише назви книг.
Використовуй Українську мову для відповіді.

[/INST]

[INST]
Жанр книги:{жанр}
Формат відповіді:{format_instructions}
Відповідь:

[/INST]

""",
    partial_variables={'format_instructions': format_instructions})

chain2 = prompt | llm | parser

user_input = input("Вкажіть назву книги:")

data = {'назва': user_input }

response1 = chain1.invoke(data)
print(response1)

response2 = chain2.invoke(response1)
print(response2)

"Напишіть модель для генерації листа:
# Перший ланцюг отримує короткий опис листа та
#генерує основний зміст
# Другий ланцюг отримує основний зміст та стиль
#листа(формальний, неформальний, тощо) та генерує
#лист"





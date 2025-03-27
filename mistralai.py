import dotenv

import os

# dotenv.load_dotenv()
# res = os.getenv('HUGGINGFACEHUB_API_TOKEN')

from  langchain_huggingface import HuggingFaceEndpoint

dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    # top_k = 3, # вибрати серед трьоз найймовірніших слів
    # top_p = 0.6, # вибрати серед слів, сума ймовірностей яких дорівнює 60%
    temperature=0.7,
    # # низька т-ра < 0.3 - мала креативність, формальні але чіткі відповіді
    # # висока т-ра > 0.6 - креативність, більш живі відповіді, але менш надійні та чіткі
    # # велика т-ра > 1,2 - галюцинації
    # max_new_tokens = 50 # макимальна довжина відповіді
)
# інструкції в mistral
# response = llm.invoke('[INST]Столиця Франції? Дай відповідь одним словом[/INST]')
# print(response)
# response_2 = llm.invoke('Столиця Франції? Дай відповідь одним словом')
# print(response_2)

#


response = llm.invoke('[INST]Напиши казку про промтінженера. Має бути 5 речень[/INST]')
print(response)

print('-------------------------------------')

response_2 = llm.invoke('Напиши казку про промтінженера. Має бути 5 речень')
print(response_2)
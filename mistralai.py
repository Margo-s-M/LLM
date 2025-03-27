import dotenv

import os

# dotenv.load_dotenv()
# res = os.getenv('HUGGINGFACEHUB_API_TOKEN')

from  langchain_huggingface import HuggingFaceEndpoint

dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    #top_k = 3,
    #top_p = 0.8,# сума ймовірностей сума яких 80%
    temperature = 0.4, #низька температура 0,3= мала креативність але чіткі відповіді>0.6креативна
    max_new_token = 50 #максимальна довжина відповіді



)

response = llm.invoke('[INST]чій крим?[]/INST')
print(response)


from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain

import tiktoken

import openai

openai.debug = True
openai.log = "debug"


loader = PyPDFLoader('data/lian.pdf')
docs = loader.load()

print(len(docs))


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
print('token size ', len(encoding.encode('hello world')))
doc_content = ''

for doc in docs:
    doc_content += doc.page_content

print('token size ', len(encoding.encode(doc_content)))


system_prompt = """你是一个专业的保险经纪人，我会给你一段保险文字，你从这些文字中提炼出足够多的问题和答案对来帮助客户更好的理解这些文字。
问题和答案对格式如下：
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

在 ``` 中间的文字必须是json格式.
"""

query = """请从如下的内容中提炼出足够多的问答对，内容是是 json 格式，内容如下:
----------------
{text}"""

response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query.format(text=doc_content)},
    ],
    model='gpt-3.5-turbo-16k',
    temperature=0,
)

print(response['choices'][0]['message']['content'])

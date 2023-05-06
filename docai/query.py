

import gradio as gr
from langchain.vectorstores import Pinecone, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import os
import pprint

from rich.console import Console
from rich.markdown import Markdown
console = Console()


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

persist_directory = "./db"

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

llm = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo', max_tokens=2000,
                 openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm)


title = """<h1 align="center"> 文档机器人</h1>"""


async def run(input):
    """
    Run the chatbot and return the response.
    """
    docs = vectordb.similarity_search(input, include_metadata=True, k=2)
    result = await chain.arun(input_documents=docs, question=input)
    return result


async def predict(input, history):

    history.append({"role": "user", "content": input})
    response = await run(input)
    history.append({"role": "assistant", "content": response})
    messages = [(history[i]["content"], history[i+1]["content"])
                for i in range(0, len(history)-1, 2)]
    return messages, history, ''

with gr.Blocks(theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_sm, text_size=gr.themes.sizes.text_sm)) as demo:

    gr.HTML(title)
    chatbot = gr.Chatbot(label="TRTC Chatbot",
                         elem_id="chatbox").style(height=800)
    state = gr.State([])

    txt = gr.Textbox(show_label=False,
                     placeholder='输入问题，比如“UserSig如何计算？”, 然后回车')

    txt.submit(predict, [txt, state], [chatbot, state, txt])

demo.queue(concurrency_count=20)
demo.launch(server_port=8080, inbrowser=True, share=True)

import os
import streamlit as st
# import torch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

model_class = 'hf_endpoint'
api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

st.set_page_config(page_title='Virtual Assistant 🤖', page_icon='🤖')
st.title('Virtual Assistant 🤖')

def model_hf(model='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.1):
  llm_endpoint = HuggingFaceEndpoint(repo_id=model,
                                     'temperature':temperature,
                                     'return_full_text': False,
                                     'max_new_tokens': 512,
                                     )
  llm_chat = ChatHuggingFace(llm=llm_endpoint)
  return llm_chat

def model_openai(model='gpt-4o-mini', temperature=0.1):
  llm_chat = ChatOpenAI(model=model, temperature=temperature)
  return llm_chat

def model_ollama(model='phi-3', temperature=0.1):
  llm_chat = ChatOllama(model=model, temperature=temperature)
  return llm_chat

def model_response(user_query, chat_history, model_class):
  # carrega modelo
  if model_class == 'hf_endpoint':
    llm_chat = model_hf()
  elif model_class == 'openai':
    llm_chat = model_openai()
  elif model_class == 'ollama':
    llm_chat = model_ollama()

  # definição dos prompts
  system_prompt = """You are a helpful assistant answering general questions.CRITICAL: Always reason and plan in English, but you MUST write your Final Answer to the user in Brazilian Portuguese."""

  if model_class.startswith('hf'):
    user_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>Você é um assistente virtual prestativo e está respondendo perguntas gerais<|eot_id|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
  else:
    user_prompt = "{input}"

  st.write(user_query)
  st.write(chat_history)
  st.write(user_prompt)

  prompt_template = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),
          MessagesPlaceholder(variable_name="chat_history"),
          ("user", user_prompt),
      ]
  )

  # criação da chain
  chain = prompt_template | llm_chat | StrOutputParser()

  return chain.stream({"chat_history": chat_history, "input": user_query})

# check de sessão do chat
if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content='Oi, sou seu assistente virtual! Como poss ajudar você ?')]

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message('AI'):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message('Human'):
      st.write(message.content)

user_query = st.chat_input('Digite sua mensagem aqui...')

if user_query is not None and user_query.strip()!='':
  st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message('Human'):
    st.markdown(user_query)

  with st.chat_message('AI'):
    resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))
        
  st.session_state.chat_history.append(AIMessage(content=resp))

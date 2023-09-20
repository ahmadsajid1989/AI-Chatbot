import os
import re
from abc import ABC

import streamlit as st
from auto_gptq import AutoGPTQForCausalLM
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage
from langchain.vectorstores import Qdrant
from qdrant_client import models, QdrantClient
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, pipeline

from htmlTemplates import css, user_template, bot_template

load_dotenv()

DEVICE = "cuda:0"
model_name_or_path = "TheBloke/Nous-Hermes-13B-GPTQ"

client = QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"))

vector_config = models.VectorParams(
    size=768,
    distance=models.Distance.COSINE
)

DEFAULT_TEMPLATE = """
### Instruction:
You are a customer support agent for Bongo, an OTT platform. Based on the chat history and the provided context, answer the user's questions. If this is your first response, begin by asking for the user's subscribed mobile phone number or email address. Always remember:
- Digital refunds are not possible.
- Keep answers concise, compassionate, and informative.
- only excepts english input otherwise say you can not respond
- If unsure, say so and direct the user to email support@bongobd.com.

{context}
{chat_history}
### Input: {question}
### Response:
""".strip()


class Chatbot:
    def __init__(self, text_pipeline: HuggingFacePipeline, embeddings: HuggingFaceEmbeddings,
                 prompt_template: str = DEFAULT_TEMPLATE,
                 verbose: bool = False):
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"], template=prompt_template
        )

        self.chain = _create_chain(text_pipeline, prompt, verbose)

        self.db = Qdrant(
            client=client, collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            embeddings=embeddings)

    def __call__(self, user_input: str, callback=None) -> str:
        docs = self.db.similarity_search(user_input)
        response = self.chain.run({'input_documents': docs, "question": user_input})

        if callback:
            for token in response.split():
                callback.on_llm_new_token(token)

        return response


def get_model():
    use_triton = False
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        use_safetensors=True,
        trust_remote_code=True,
        device=DEVICE,
        use_triton=use_triton,
        uantize_config=None,
        max_memory={0: "15360MiB", 1: "15360MiB", 2: "15360MiB", 3: "15360MiB"},
    )

    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return tokenizer


def get_generation_config():
    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    return generation_config


def get_embedding():
    model_name = "embaas/sentence-transformers-multilingual-e5-base"
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings


def _create_chain(text_pipeline: HuggingFacePipeline, prompt: PromptTemplate, verbose: bool = False):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="### Input",
        ai_prefix="### Response",
        input_key="question",
        output_key="output_text",
        return_messages=False
    )

    return load_qa_chain(
        text_pipeline,
        chain_type="stuff",
        prompt=prompt,
        memory=memory,
        verbose=verbose
    )


def get_pipeline():
    tokenizer = get_tokenizer()
    model = get_model()
    generation_config = get_generation_config()
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, use_multiprocessing=False
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
        streamer=streamer,
        batch_size=1
    )

    return pipe


# def main():
#     st.set_page_config(page_title="Bongo Bot",
#                        page_icon=":male-office-worker:")
#     st.write(css, unsafe_allow_html=True)
#
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
#
#     st.header("Bongo Bot :male-office-worker:")
#
#     user_question = st.text_input("How can I help you today?")
#
#     if user_question:
#         llm = HuggingFacePipeline(pipeline=get_pipeline())
#
#         chatbot = Chatbot(text_pipeline=llm, embeddings=get_embedding(), verbose=False)
#         st.write(chatbot(user_question))
#
#
# if __name__ == '__main__':
#     main()

class StreamHandler(BaseCallbackHandler, ABC):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    def on_response_complete(self):
        formatted_text = format_text(self.text)
        self.container.markdown(formatted_text)


def format_text(token):
    # Add spaces before capital letters followed by lowercase letters
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', token)



def main():
    st.set_page_config(page_title="Bongo Bot", page_icon=":male-office-worker:")
    st.write(css, unsafe_allow_html=True)

    if "chatbot" not in st.session_state:
        llm = HuggingFacePipeline(pipeline=get_pipeline())
        st.session_state.chatbot = Chatbot(text_pipeline=llm, embeddings=get_embedding(), verbose=False)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Bongo Bot :male-office-worker:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = st.session_state.chatbot(prompt, callback=stream_handler)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))


if __name__ == '__main__':
    main()

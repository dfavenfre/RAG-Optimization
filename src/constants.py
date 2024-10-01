# langchain
from langchain_openai import (ChatOpenAI, OpenAIEmbeddings)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_cohere import CohereEmbeddings

# openai
from openai import OpenAI

# Cohere
import cohere

import os

openai_turbo = ChatOpenAI(
    openai_api_key=os.environ.get("openai_api_key"),
    temperature=1e-10,
    max_tokens=1000,
    model_name="gpt-3.5-turbo"
)

evaluator = ChatOpenAI(
    openai_api_key=os.environ.get("openai_api_key"),
    temperature=1e-10,
    max_tokens=1000,
    model_name="gpt-4o"
)

bge_m3 = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

ada_002 = OpenAIEmbeddings(
    openai_api_key=os.environ.get("cohere_api_key"),
)

cohere_embedding = CohereEmbeddings(
    cohere_api_key=os.environ.get("cohere_api_key"),
    model="embed-multilingual-v3.0"
)

client = OpenAI(api_key=os.environ.get("openai_api_key"))
cohere_client = cohere.Client(api_key=os.environ.get("cohere_api_key"))
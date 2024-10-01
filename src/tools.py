from operator import itemgetter
from typing import Optional, Union

from langchain.output_parsers import PydanticOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from transformers import pipeline

from src.constants import openai_turbo
from src.helper_functions import TopicIdentifier, format_documents, filter_embeddings, get_unique_union, \
    apply_reciprocal_ranking


def chunk_summarizer(
        raw_prompt: str,
        chunk_context: str,
        llm: Optional[Union[ChatOpenAI, HuggingFacePipeline, pipeline]] = openai_turbo
) -> str:
    "Summarizes text context and returns a list of keywords based on the given context"

    topic_parser = PydanticOutputParser(pydantic_object=TopicIdentifier)
    system_prompt = PromptTemplate(
        template=raw_prompt,
        input_variables=["context"],
        partial_variables={"output_format": topic_parser.get_format_instructions()}
    )

    summarize_chain = (
            {
                "context": itemgetter("context")
            }
            | system_prompt
            | llm
            | topic_parser
    )

    return summarize_chain.invoke({"context": chunk_context})


def stuff_method(
        retriever_object,
        question,
        llm: Optional[Union[ChatOpenAI, HuggingFacePipeline, pipeline]] = openai_turbo
):
    stuff_chain = (
            {
                "context": itemgetter("question")
                           | retriever_object.as_retriever()
                           | format_documents,
                "question": itemgetter("question")
            }
            | rag_prompt
            | llm
            | StrOutputParser()
    )
    return stuff_chain.invoke({"question": question})


def contextual_compression(
        retriever_object,
        embedding_funtion,
        question,
        llm: Optional[Union[ChatOpenAI, HuggingFacePipeline, pipeline]] = openai_turbo
):
    compression_retriever = filter_embeddings(
        retriever_object,
        embedding_funtion
    )
    contextual_compression_chain = (
            {
                "context": itemgetter("question")
                           | compression_retriever
                           | format_documents,
                "question": itemgetter("question")
            }
            | rag_prompt
            | llm
            | StrOutputParser()

    )

    return contextual_compression_chain.invoke({"question": question})


def multi_query(
        question: str,
        retriever_object
):
    multi_query_generator = (
            multi_query_prompt
            | openai_turbo
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    )

    doc_retriever_chain = (
            multi_query_generator
            | retriever_object.as_retriever().map()
            | get_unique_union
    )

    rag_chain = (
            {
                "context": doc_retriever_chain,
                "question": itemgetter("question")
            }
            | rag_prompt
            | openai_turbo
            | StrOutputParser()
    )

    return rag_chain.invoke({"question": question})


def step_down(
        question: str,
        retriever_object: FAISS
) -> str:
    query_step_down_generator = (
            {
                "question": itemgetter("question")
            }
            | stepdown_fs_prompt
            | openai_turbo
            | StrOutputParser()
    )

    stepdown_retrieval_chain = (
            {
                "normal_context": RunnableLambda(lambda x: x["question"]) | retriever_object.as_retriever(),
                "step_back_context": query_step_down_generator | retriever_object.as_retriever(),
                "question": query_step_down_generator,
            }
            | stepdown_system_prompt
            | openai_turbo
            | StrOutputParser()
    )

    return stepdown_retrieval_chain.invoke({"question": question})


def reciprocal_rag(
        question: str,
        retriever_object: FAISS,
        n_documents: Optional[int] = 8
):
    generate_queries = (
            {
                "question": itemgetter("question")
            }
            | rag_fusion_system_prompt
            | openai_turbo
            | StrOutputParser()
            | (lambda x: x.split("\n"))

    )

    retrieval_chain_rag_fusion = (
            generate_queries
            | retriever_object.as_retriever(k=n_documents, fetch_k=n_documents).map()
            | apply_reciprocal_ranking
    )

    rag_chain = (
            {
                "context": retrieval_chain_rag_fusion,
                "question": itemgetter("question")
            }
            | rag_prompt
            | openai_turbo
            | StrOutputParser()
    )

    return rag_chain.invoke({"question": question})

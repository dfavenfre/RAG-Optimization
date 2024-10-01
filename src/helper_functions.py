# langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain.load import dumps, loads

# scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# constants
from constants import *

# basics
from typing import Optional, Tuple, Dict, Union, List, Callable
from pydantic import BaseModel, Field
import plotly.express as px
import numpy as np
import time
import psutil
import os


class TopicIdentifier(BaseModel):
    keywords: List[str] = Field(
        description="Identify the topics based on the question titles in the given context and put all topics in a list"
    )


class IdentifyQuestionTopic(BaseModel):
    keywords: str = Field(
        description="Identify the topic as a keyword based on the question in the given context."
    )


def chunk_up_documents(
        file_path: str,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: Optional[int] = 100
):
    documents = []
    for file in os.listdir(file_path):
        if file.endswith(".pdf"):
            pdf_path = file_path + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


def format_documents(docs: List[str]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_openai_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_cohere_embeddings(text):
    response = cohere_client.embed(texts=[text])
    embedding = response.embeddings[0]
    return embedding


def get_bge_embeddings(text):
    embeddings = bge_m3.embed_query(text)
    return embeddings


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def filter_embeddings(
        search_object,
        embedding_model,
        s_threshold: Optional[int] = 0.7,
        r_threshold: Optional[int] = 0.7
):
    relevancy_filter = EmbeddingsFilter(
        embeddings=embedding_model,
        s_threshold=s_threshold
    )
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=embedding_model,
        r_threshold=r_threshold
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevancy_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=search_object.as_retriever()
    )

    return compression_retriever


def measure_resource_usage(function: Callable, **parameters):
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=5)
    ram_before = psutil.virtual_memory().percent

    query_result = function(**parameters)
    time.sleep(2)

    cpu_after = psutil.cpu_percent(interval=5)
    ram_after = psutil.virtual_memory().percent
    end_time = time.time()
    runtime = np.abs(end_time - start_time)

    cpu_usage_percentage = np.abs(round(cpu_after, 4) - round(cpu_before, 4))
    ram_usage_percentage = np.abs(round(ram_after, 4) - round(ram_before, 4))

    return runtime, cpu_usage_percentage, ram_usage_percentage, query_result


def apply_reciprocal_ranking(results: list[list], k=60):
    """
    Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula.
    """

    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results[:4]


def combine_embeddings(
        query: str,
        vector_store: FAISS,
        embedding_model: Union['OpenAIEmbeddings', 'CohereEmbeddings', 'HuggingFaceBgeEmbeddings'],
        number_of_documents: Optional[int] = 4,
        calculate_similarity: Optional[bool] = True
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[float]]]:
    similarity_scores: List[np.ndarray] = []
    sanitized_document_content: List[str] = []

    query_embedding = embedding_model.embed_query(query)
    rag_embeddings = vector_store.similarity_search(query, k=number_of_documents)

    for doc in rag_embeddings:
        sanitized_document_content.append(doc.page_content)

    rag_vector_embeddings = embedding_model.embed_documents(sanitized_document_content)

    for doc_embedding in rag_vector_embeddings:
        similarity_scores.append(cosine_similarity([query_embedding], [doc_embedding])[0][0])

    if calculate_similarity:
        return query_embedding, rag_vector_embeddings, similarity_scores

    return query_embedding, rag_vector_embeddings


def plotly_plot_2d_embeddings(embeddings_2d, method_name, labels, similarity_scores=None):
    if similarity_scores:
        hover_text = ['Query'] + [f'Similarity: {score:.2f}' for score in similarity_scores]
    else:
        hover_text = labels

    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=labels,
        title=f'{method_name} 2D Visualization of Query and Retrieved Embeddings',
        labels={'color': 'Embeddings'},
        hover_name=hover_text
    )
    fig.show()


def visualize_embeddings(query_embedding: np.ndarray, retrieved_embeddings: np.ndarray, similarity_scores: List[float]):
    all_embeddings = np.vstack([query_embedding, retrieved_embeddings])
    tsne = TSNE(n_components=2, perplexity=min(len(retrieved_embeddings) - 1, 30), random_state=42)
    all_embeddings_2d_tsne = tsne.fit_transform(all_embeddings)
    labels = ['Query'] + ['Vector'] * len(retrieved_embeddings)
    plotly_plot_2d_embeddings(all_embeddings_2d_tsne, 't-SNE', labels, similarity_scores)


def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.5f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')


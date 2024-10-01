# RAG Optimization

## Dataset
[Dataset](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/data/documents/rag_documents.zip) comprises 12 publicly available documents related to insurance policies and campaigns from [Sigortam.net](https://www.sigortam.net/). These documents present various contextual challenges, including some that contain numeric values associated with insurance costs and fees based on different vehicle information. Additionally, several documents include unstructured tables, which can complicate the retrieval and generation of accurate information in RAG-based applications.

## Methods 

The `chunk_up_documents` function is designed to process PDF documents in a specified directory, chunking their text into smaller, manageable segments.

* File Reading: The function iterates through all files in the given file_path, checking for PDF files. It uses PyPDFLoader to load the content of each PDF file and appends the loaded documents to a list.

* Text Splitting: A RecursiveCharacterTextSplitter is initialized with specified parameters: chunk_size, which defines the maximum size of each text chunk, and chunk_overlap, which determines how much text from the end of one chunk overlaps with the beginning of the next. The splitting is done using defined separators (in this case, double newlines).

* Returning Chunks: Finally, the function returns a list of chunked documents, allowing for further processing or analysis.

### Document Preprocessing
```Python
# Chunking Methodology
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

# Example Usage
document_chunks = chunk_up_documents(
    file_path="/content/rag_documents/",
    chunk_size=2000,
    chunk_overlap=0
)
print(len(document_chunks))
```
### Vectorstore
FAISS had been used as the vectorstore across various optimization processes / comparison. 

```Python
# Example Usage
faissdb_cohere = FAISS.from_documents(document_chunks, cohere_embedding)
faissdb_cohere.save_local("faiss_cohere")
```

### Grid-search Optimization 
RAG (Retrieval-Augmented Generation) optimization was performed, within a grid-search, using various methodologies— [Stuff](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/tools.py#L44), [Query Step-Down](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/tools.py#L119), [Multi-Query](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/tools.py#L89), [Contextual Compression](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/tools.py#L63), and [Reciprocal](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/tools.py#L146) —across different embedding models, including ada-002 (OpenAI), cohere-v3-multilingual (Cohere), and bge-en-small (BGE). A total of 378,286 tokens (including both prompt and completion) were processed to determine which RAG method and embedding model combination yielded the highest accuracy. 

The performance comparison was based on the evaluation dataset available [here](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/data/evaluation_dataset/rag_reference_data.xlsx), assessed by GPT-4, with answers generated using GPT-3.5-Turbo-0125. The evaluation focused on several LLM-based metrics, including Coherence, Conciseness, Contextual Accuracy, Helpfulness, and Relevance. To see detailed [LangEval results](https://smith.langchain.com/public/f6bfe72d-262b-4596-807d-1eef5016450e/d)
 

# Evaluation & Results
The [test dataset](https://github.com/dfavenfre/RAG-Optimization/blob/main/src/data/evaluation_dataset/rag_reference_data.xlsx) consists of frequently asked questions sourced from the Sigortam.net website. Q&A pairs include question and answers relevant to ad-campaigns, promotions, as well as numerical values (fees, charges, etc.) that are essential for the accurate generation of responses in a RAG (Retrieval-Augmented Generation) system.

## Number of Correct Answers Out of 12
| Embedding | RAG Method               | Coherence | Conciseness | Cot Contextual Accuracy | Relevance | Helpfulness |
|-----------|--------------------------|-----------|-------------|-------------------------|-----------|-------------|
| OpenAI    | Step-Down                 | 10.0      | 3.0         | 6.0                     | 6.0       | 8.0         |
| BGE       | Step-Down                 | 12.0      | 4.0         | 8.0                     | 7.0       | 11.0        |
| Cohere    | Step-Down                 | 11.0      | 3.0         | 8.0                     | 7.0       | 9.0         |
| BGE       | Multi-Query               | 12.0      | 5.0         | 9.0                     | 8.0       | 12.0        |
| Cohere    | Reciprocal                | 11.0      | 7.0         | 9.0                     | 9.0       | 11.0        |
| BGE       | Stuff Method              | 11.0      | 7.0         | 10.0                    | 10.0      | 10.0        |
| Cohere    | Multi-Query               | 12.0      | 7.0         | 10.0                    | 9.0       | 10.0        |
| BGE       | Reciprocal                | 12.0      | 6.0         | 10.0                    | 10.0      | 11.0        |
| OpenAI    | Stuff Method              | 11.0      | 7.0         | 10.0                    | 7.0       | 11.0        |
|           | Multi-Query               | 12.0      | 5.0         | 11.0                    | 8.0       | 11.0        |
|           | Reciprocal                | 12.0      | 6.0         | 11.0                    | 8.0       | 12.0        |
| BGE       | Contextual Compression    | 12.0      | 7.0         | 12.0                    | 10.0      | 12.0        |
| Cohere    | Contextual Compression    | 12.0      | 6.0         | 12.0                    | 9.0       | 12.0        |
|           | Stuff Method              | 12.0      | 8.0         | 12.0                    | 10.0      | 11.0        |
| OpenAI    | Contextual Compression    | 12.0      | 9.0         | 12.0                    | 9.0       | 12.0        |

## Latency & Error Rate by Embedding models & RAG Methods
| Embedding | RAG Method               | P50 Latency | P99 Latency | Error Rate |
|-----------|--------------------------|-------------|-------------|------------|
| BGE       | Stuff Method              | 2.42        | 4.55        | 0.0        |
| OpenAI    | Stuff Method              | 2.53        | 4.48        | 0.0        |
| Cohere    | Stuff Method              | 2.68        | 4.97        | 0.0        |
| OpenAI    | Contextual Compression    | 2.71        | 4.75        | 0.0        |
| Cohere    | Contextual Compression    | 3.58        | 6.11        | 0.0        |
| OpenAI    | Reciprocal                | 3.78        | 7.15        | 0.0        |
|           | Multi-Query               | 3.91        | 5.81        | 0.0        |
| Cohere    | Multi-Query               | 4.72        | 12.62       | 0.0        |
|           | Reciprocal                | 4.77        | 9.90        | 0.0        |
| BGE       | Reciprocal                | 5.25        | 13.57       | 0.0        |
|           | Multi-Query               | 5.83        | 14.78       | 0.0        |
| OpenAI    | Step-Down                 | 6.54        | 15.80       | 17.0       |
| Cohere    | Step-Down                 | 6.79        | 10.38       | 0.0        |
| BGE       | Step-Down                 | 7.28        | 12.48       | 0.0        |
|           | Contextual Compression    | 31.08       | 40.42       | 0.0        |


## Performance of RAG Methods by Coherence, CoT Context Accuracy, Conciseness and Relevancy scores
![image](https://github.com/user-attachments/assets/7b0e1814-db08-48f2-80b6-c7d76bc3b69c)

## Performance of RAG Methods by Total Cost, Latency, Completion Token, Prompt Token usages
![image](https://github.com/user-attachments/assets/f6335a8d-5c24-4176-9c6b-ae36d3cedc14)

## Performance of Embedding & RAG Methods by P50, P99 Latencies and Error Rate %
![image](https://github.com/user-attachments/assets/25b123ae-0446-496e-945b-791c4658be48)



import os
import time
import torch

from huggingface_hub import login, whoami
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
### WORKAROUND for "trust_remote_code=True is required" error in HuggingFaceEmbeddings()
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv(verbose=True)
cache_dir         = os.environ['CACHE_DIR']
model_vendor      = os.environ['MODEL_VENDOR']
model_name        = os.environ['MODEL_NAME']
model_precision   = os.environ['MODEL_PRECISION']
inference_device  = os.environ['INFERENCE_DEVICE']
vectorstore_dir   = os.environ['VECTOR_DB_DIR']
num_max_tokens    = int(os.environ['NUM_MAX_TOKENS'])
embeddings_model  = os.environ['MODEL_EMBEDDINGS']
rag_chain_type    = os.environ['RAG_CHAIN_TYPE']
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":cache_dir}

hf_token = os.getenv('HUGGINGFACE_TOKEN')

try:
    whoami()
    print('Authorization token already provided')
except OSError:
    print('The llama2 model is a controlled model.')
    print('You need to login to HuggingFace hub to download the model.')
    login()
finally:
    model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True, cache_dir=cache_dir)
    embeddings = HuggingFaceEmbeddings(
        model_name = embeddings_model,
        model_kwargs = {'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 1})
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.5})

    model_id = f'{model_vendor}/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    llm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

def evaluate_relevance(documents, query):
    # Placeholder function to evaluate relevance.
    # This should be replaced with your actual relevance criteria.
    relevant_docs = [doc for doc in documents if is_relevant(doc, query)]
    return relevant_docs

def is_relevant(doc, query):
    # Example relevance check: if the query appears in the document
    return query.lower() in doc.page_content.lower()

# Function to generate responses
def generate_response(query: str):
    retrieved_documents = retriever.get_relevant_documents(query)
    print(f'retrieved docs: {retrieved_documents}')
    
    relevant_documents = evaluate_relevance(retrieved_documents, query)
    precision = len(relevant_documents) / len(retrieved_documents) if retrieved_documents else 0
    print(f'precision: {precision}')

    context = " ".join([doc.page_content for doc in relevant_documents])

    query_tokens = tokenizer(query, return_tensors='pt', max_length=num_max_tokens//2, truncation=True)
    context_tokens = tokenizer(context, return_tensors='pt', max_length=num_max_tokens//2, truncation=True)

    inputs = {key: torch.cat([query_tokens[key], context_tokens[key]], dim=-1) for key in query_tokens.keys()}

    outputs = llm_model.generate(**inputs, max_length=num_max_tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


app = FastAPI()

@app.get('/chatbot/{item_id}')
async def root(item_id:int, query:str=None):
    if query:
        stime = time.time()
        ans = generate_response(query)
        etime = time.time()
        process_time = etime - stime
        return {'response': f'{ans} \r\n\r\n 处理时间: {process_time:6.1f} 秒'}
    return {'回答':''}


# API reference
# http://127.0.0.1:8000/docs

# How to run (You need to have uvicorn and streamlit -> pip install uvicorn streamlit)
# uvicorn servicemesh-rag-server:app

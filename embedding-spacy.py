import os
import glob
from dotenv import load_dotenv
from tqdm import tqdm
import pickle

from langchain_core.documents.base import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import SpacyTextSplitter

load_dotenv(verbose=True)
document_dir = os.environ['DOCUMENT_DIR']
vectorstore_dir = os.environ['VECTOR_DB_DIR']
embeddings_model = os.environ['MODEL_EMBEDDINGS']
cache_dir = os.environ['CACHE_DIR']


### WORKAROUND for "trust_remote_code=True is required" error in HuggingFaceEmbeddings()
from transformers import AutoModel
model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True, cache_dir=cache_dir) 


##
## Read HTML documents and extract main section
## Note: Designed dedicated for Service Mesh web documents
##
def generate_documents_from_data(glob_pattern:str, max_doc_count:int=-1) -> list[Document]:
    doc_count = 0

    data_files = glob.glob(glob_pattern, recursive=True)

    documents = []
    for data_file in tqdm(data_files):
        print(f'*** {data_file}')
        with open(data_file, 'rt', encoding='utf-8') as f:
            data_contents = f.read() 

        doc = Document(page_content=data_contents, metadata={'source':data_file})
        documents.append(doc)

        # Extract text from 'main' tag
    #    main_section = soup.find('main')
    #    if main_section is not None:
    #        text = main_section.get_text()
    #        text = ''.join([line+'\n' for line in text.splitlines() if line != '']) # Remove empty lines
    #        doc = Document(page_content=text, metadata={'source':data_file})
    #        documents.append(doc)

    #        doc_count += 1
    #        if max_doc_count != -1 and doc_count >= max_doc_count:
    #            break

    return documents



def generate_vectorstore_from_documents(
        docs             :list[Document],
        vectorstore_dir  :str  = './db',
        chunk_size       :int  = 300,
        chunk_overlap    :int  = 0,
        normalize_emb    :bool = False,
        embeddings_model :str  = 'sentence-transformers/all-mpnet-base-v2',
        pipeline         :str  = 'en_core_web_sm'
    ) -> None:
    print('*** Splitting documents into smaller chunks')
    print(f'Chunk size : {chunk_size}, Chunk overlap : {chunk_overlap}')
    
    if docs is None or len(docs) == 0:
        print("No documents to process.")
        return

    splitter = SpacyTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        pipeline=pipeline               # en_core_web_sm, ja_core_news_sm, etc.
    )

    splitted_docs = splitter.split_documents(docs)

    print('*** Generate embedding and storing splitted documents into vector store')
    embeddings = HuggingFaceEmbeddings(
        model_name = embeddings_model,
        model_kwargs = {'device':'cpu'},
        encode_kwargs = {'normalize_embeddings':normalize_emb}
    )

    vectorstore = Chroma(
        persist_directory=vectorstore_dir, 
        embedding_function=embeddings
    )

    for doc in tqdm(splitted_docs):
        vectorstore.add_documents([doc])




# Generate documents from HTML. Read the documents from pickle file if exists.
pickle_file = './doc_obj.pickle'
if not os.path.exists(pickle_file):
    print('*** Reading HTML and generating document(s)')
    docs = generate_documents_from_data(f'{document_dir}/**/*.md')
    with open(pickle_file, 'wb') as f:
        pickle.dump(docs, f)
else:
    print(f'*** Reading documents from a pickled file ({pickle_file})')
    with open(pickle_file, 'rb') as f:
        docs = pickle.load(f)

print('*** Converting documents into embeddings and creating a vector store(s)')
generate_vectorstore_from_documents(docs, vectorstore_dir,  500,   0, True, embeddings_model)
# generate_vectorstore_from_documents(docs, vectorstore_dir,  300,   0, True, embeddings_model)
# generate_vectorstore_from_documents(docs, vectorstore_dir,  500,   0, True, embeddings_model)
# generate_vectorstore_from_documents(docs, vectorstore_dir,  500, 100, True, embeddings_model)
# generate_vectorstore_from_documents(docs, vectorstore_dir, 1000, 100, True, embeddings_model)

# langchain-chatbot-rag
This is a simplified RAG implementation based on LLM with dynamical configuration.

## Command for repo

$ deactivate
$ rm -rf myvenv
$ python3 -m venv myvenv
$ source myvenv/bin/activate
$ python -m pip install --upgrade pip 
$ pip install wheel setuptools
$ pip install -r requirements.txt

$ python embedding-md.py
$ uvicorn rag-server:app --host 0.0.0.0
$ streamlit run rag-client.py

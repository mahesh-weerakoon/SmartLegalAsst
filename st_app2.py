import streamlit as st
# LangChain components to use
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import  OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from streamlit_extras.add_vertical_space import add_vertical_space

#side bar
with st.sidebar:
    st.title('Smart Legal Assistant')
    st.markdown('''
    - [Main Menu](http://google.com)
    - [Upload PDF] ()
''')
    add_vertical_space(5)
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        st.write(pdf)
        #pdf_rr = PdfReader(pdf)
    st.write('Smart Legal Assistant - POC version')

ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

pdfreader = PdfReader('explanation.pdf')

# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="tbl_legal_doc",
    session=None,
    keyspace=None,
)



text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

astra_vector_store.add_texts(texts[:50])

print("Inserted %i headlines." % len(texts[:50]))
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


st.header('Smart Legal Assistant')

if "messages" not in st.session_state:
    st.session_state.messages =[]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt:= st.chat_input("Message"):
    msg = {
        'role':'user',
        'content':prompt        
    }
    st.session_state.messages.append(msg)

    with st.chat_message('user'):
        st.markdown(prompt)

    

    with st.chat_message('assistant'):
        chatresponse = astra_vector_index.query(prompt, llm=llm).strip()
        st.markdown(chatresponse)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":chatresponse
        }
    )
    
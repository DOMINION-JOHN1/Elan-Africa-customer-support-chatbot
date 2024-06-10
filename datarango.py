import streamlit as st
import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

# Set environment variables
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize model and embeddings
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
parser = StrOutputParser()

# Define the prompt template
template = """
Answer the questions based on the context below. If you can't
answer the question, reply "I don't have such information in my database". Also answer expertly any code related question you are asked.

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

# Load documents
loader = WebBaseLoader("https://elanafrica.com/")
pages = loader.load_and_split()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = text_splitter.split_documents(pages)

# Create Pinecone Vector Store
index_name = "datarango"
pinecone = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

# Define the chain
chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Streamlit app setup
st.title("Chat with elanAfrica customer support bot")
st.write("Ask any question about elanAfrica today :earth_africa:.")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to get response from the bot
def get_response(user_input):
    question = user_input
    response = chain.invoke(question)
    return response

# Text input for user question
user_input = st.text_input("You: ", key="input")

# If user submits a question
if user_input:
    response = get_response(user_input)
    st.session_state.history.append({"user": user_input, "bot": response})

# Display chat 
for chat in st.session_state.history:
    st.write(f"You: {chat['user']}")
    st.write(f"Bot: {chat['bot']}")

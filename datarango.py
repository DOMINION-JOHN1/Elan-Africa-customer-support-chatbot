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
from langchain_core.documents import Document
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup

# Set environment variables
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize model and embeddings
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
parser = StrOutputParser()

# Define the prompt template
template = """
Derive all your answers based on the information in the context. Be creative, don't just recommend pages on the website.
Write out the answer to the question directly to the user instead of directing them to the site. 
Ensure you know about Elan Africa, its vision, mission, and what it does generally. 
Even if you don't know the answer immediately, go through the context again and provide the most accurate answer and suggestion. 

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

# Function to extract text from PDF
"""
def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to fetch and parse web pages
def fetch_all_pages(url):
    pages_content = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Add main page content
    pages_content.append(Document(page_content=soup.get_text(), metadata={"url": url}))
    
    # Find all links to other pages/tabs
    links = soup.find_all('a', href=True)
    for link in links:
        href = link['href']
        if href.startswith('/'):
            href = url.rstrip('/') + href
        if not href.startswith('http'):
            continue
        
        if href.endswith('.pdf'):
            # Handle PDF links
            pdf_text = extract_text_from_pdf(href)
            pages_content.append(Document(page_content=pdf_text, metadata={"url": href}))
        else:
            # Handle regular HTML links
            sub_response = requests.get(href)
            sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
            pages_content.append(Document(page_content=sub_soup.get_text(), metadata={"url": href}))
    
    return pages_content


# Load local PDF and add its contents to pages_content
local_pdf_path = "Elan-Profile.pdf"
try:
    pdf_loader = PyPDFLoader(local_pdf_path)
    pdf_documents = pdf_loader.load()
    pages_content.extend(pdf_documents)
except Exception as e:
    print(f"Failed to load local PDF from {local_pdf_path}: {e}")

local_pdf_path = "Elan-Service-Packages.pdf"
try:
    pdf_loader = PyPDFLoader(local_pdf_path)
    pdf_documents = pdf_loader.load()
    pages_content.extend(pdf_documents)
except Exception as e:
    print(f"Failed to load local PDF from {local_pdf_path}: {e}")


# Scrape all pages and tabs from the website
pages_content = fetch_all_pages("https://elanafrica.com/")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = text_splitter.split_documents(pages_content)
"""
# Create Pinecone Vector Store
index_name = "elanrag"
pinecone = PineconeVectorStore(index_name=index_name, embedding=embeddings)

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

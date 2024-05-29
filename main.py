import os
import streamlit as st
from pypdf import PdfReader
from src.utils.config import DOCUMENTS_PATH, stop_words
from src.utils.llm import get_llm_model
from src.utils.prompt import get_answer_question_from_content, get_questions_from_content
import re
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import asyncio

client = chromadb.PersistentClient(path="database")
collection = client.get_or_create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}
)

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
llm = get_llm_model()

# Function to clean content
def clean_content(content: str):
    content = content.strip()  # Remove leading and trailing whitespace
    content = content.replace("\n", " ")  # Remove newlines
    return " ".join(content.split())

# Function to get PDF files in a directory
def get_pdf_files_in_directory(directory_path):
    all_files = os.listdir(directory_path)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
    return pdf_files

# Function to extract questions from prompt result
def extract_questions_from_prompt_result(text):
    text = text.replace("\n", "")
    text = text.replace("`", "")
    questions = re.split(r'\d+\.\s', text)
    if questions and questions[0] == '':
        questions.pop(0)
    return questions

# Function to get list of questions from content
def get_list_questions_from_content(content, nbr_questions=30):
    prompt = get_questions_from_content(content, nbr_questions=nbr_questions, model="mistral")
    result = llm.create_completion(
        prompt,
        stream=False,
        max_tokens=2048,
        stop=stop_words,
        temperature=0,
    )
    result = result["choices"][0]["text"]
    result = result.replace("in the context", "")
    result = result.replace("in the given context", "")
    result = clean_content(result)
    content_questions = extract_questions_from_prompt_result(result)
    return content_questions

# Function to get references for query
async def get_references_for_query(query):
    question_emb = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=[t.tolist() for t in question_emb],
        n_results=3
    )
    references = [(r["document"], r["page"]) for r in results["metadatas"][0]]
    references = sorted(list(set(references)))
    return references

# Function to process question and return response
async def process_question(question):
    references = await get_references_for_query(question)
    context = ""
    for reference in references:
        reader = PdfReader(os.path.join(DOCUMENTS_PATH, reference[0]))
        page_content = reader.pages[reference[1]].extract_text()
        context += (page_content + "\n")

    prompt = get_answer_question_from_content(context, question, model="mistral")
    stream = llm.create_completion(
        prompt,
        stream=True,
        max_tokens=4096,
        stop=stop_words,
        temperature=0.0,
    )
    result = ""
    for output in stream:
        result += output["choices"][0]["text"]
    return result

# Streamlit application
st.title("PDF Question Answering System")

# User input
question = st.text_input("Enter your question:")

# Process question if provided
if question:
    st.write("Processing...")
    with st.spinner('Processing...'):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_question(question))
        st.write(response)

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    with open(os.path.join("documents", uploaded_file.name), "wb") as buffer:
        buffer.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

# List documents
if st.button("List PDF Documents"):
    documents = get_pdf_files_in_directory(DOCUMENTS_PATH)
    st.write(documents)

# Delete document
file_to_delete = st.text_input("Enter the name of the file to delete:")
if st.button("Delete Document"):
    file_path = os.path.join(DOCUMENTS_PATH, file_to_delete)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        st.success(f"{file_to_delete} deleted successfully")
    else:
        st.error("File not found")

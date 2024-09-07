from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader 

def read_pdf(file):
    text = ""
    pdf = PdfReader(file)
    for page in pdf.pages:
        text += page.extract_text()

    return text


def get_text_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=15000,
        chunk_overlap = 1500
    )

    return splitter.split_text(texts)


def create_knowledge_base(chunks):
    print("Creating the vectordata base...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    DB_FAISS_PATH = "VectorStore/db_faiss"
    db = FAISS.from_texts(chunks,embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Done")

def user_question(question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    knowledge_db = FAISS.load_local("VectorStore/db_faiss",
                                    embeddings=embedding,
                                    allow_dangerous_deserialization=True)
    
    docs = knowledge_db.similarity_search(question)

    chain = conversational_chain()

    response = chain({
        "input_documents" :docs ,
        "question" : question
    } , return_only_outputs=True)

    return response["output_text"]


def conversational_chain():
    prompt_template = """
        Act as a helpful assistant , and answer the questions as detailed possible from the provided context and documnets, make sure
        to provide all the details, if the answer is not in provided context or document just say "answer is not available in the context"
        , be friendly in your responses but not too friendly\n\n
        Context:\n {context} \n
        Question: \n {question} \n

        Answer:
        """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.4)


    prompt = PromptTemplate(template=prompt_template , input_variable=["context" , "question"])

    chain = load_qa_chain(model,chain_type= "stuff" , prompt=prompt )

    return chain
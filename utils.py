from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain 
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from PyPDF2 import PdfReader 
from langchain_core.output_parsers import StrOutputParser
import json
from langchain.document_loaders import JSONLoader
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
import plotly.express as px
import pandas as pd
import ast

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_PROJECT']="pr-another-ruin-77"


@tool("plot_map")
def plot_map(datapoints):
    """
    Plots points on a map using latitude and longitude.
    Input: datapoints (list of dicts with 'latitude' and 'longitude').
    Example: [{"latitude": 27.321, "longitude": 34.123}]
    """
    datapoints = ast.literal_eval(datapoints)
    import plotly.express as px
    import pandas as pd

    if not all("latitude" in dp and "longitude" in dp for dp in datapoints):
        raise ValueError("Each dictionary in datapoints must have 'latitude' and 'longitude' keys.")

    # Convert datapoints into a DataFrame
    df = pd.DataFrame(datapoints)

    # Create the plot
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        zoom=5,
        mapbox_style="carto-positron"
    )
    fig.show()
    return "Map successfully plotted."


tools = [
    Tool(
        name="Plot Map",
        func=plot_map,
        description="Use this to plot datapoints on a map. Input should be a list of dictionaries with latitude and longitude."
    )
]


def read_pdf(file):
    text = ""
    pdf = PdfReader(file)
    for page in pdf.pages:
        text += page.extract_text()

    return text



def parse_json(data, parent_key=''):
    """
    Recursively parse JSON data, handling nested objects and lists.
    Convert them into a readable string format for embedding.
    """
    text = ""
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            text += parse_json(value, new_key)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            text += parse_json(item, f"{parent_key}[{index}]")
    else:
        # Base case: Append scalar values with context
        text += f"{parent_key}: {data}\n"
    return text

def read_json(uploaded_file):
    """
    Reads a JSON file from an UploadedFile object and extracts its content.
    """
    text = ""
    if uploaded_file is not None:
        data = json.load(uploaded_file)  # Parse the JSON
        text = parse_json(data)  # Recursively extract content

    return text


def get_text_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=25_000,
        chunk_overlap = 1500
    )

    return splitter.split_text(texts)


def create_knowledge_base(chunks):
    # print("Creating the vectordata base...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    DB_FAISS_PATH = "VectorStore/db_faiss"
    db = FAISS.from_texts(chunks,embeddings)
    db.save_local(DB_FAISS_PATH)
    # print("Done")

def user_question(question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    knowledge_db = FAISS.load_local("VectorStore/db_faiss",
                                    embeddings=embedding,
                                    allow_dangerous_deserialization=True)
    
    print("knowledge base created")
    docs = knowledge_db.similarity_search(question) #ab
    retreiver = knowledge_db.as_retriever() #ab
    results = generate_questiions(question)
    

    chain = conversational_chain(retreiver) #ab

    response = chain({  #ab
        "input_documents" :docs ,
        "question" : question
    } , return_only_outputs=True)
    
    # print(results)
    return response["output_text"]

def remove_empty_strings(list_of_strings):
    return [x for x in list_of_strings if x!= '' and x != ""]

def generate_questiions(question):
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002")

    chain = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
        | remove_empty_strings
        )
    results = chain.invoke({"question": question})



    return results


def conversational_chain(retreiver):
    # prompt_template = """
    #     You are a Biodiversity bot that is an expert in Marine Species, especially those in the Red Sea. 
    #     You answer questions only if the answer is in your documents,
    #     You should first extract the name of the species in the query if exists , then look for the attribute related to this species aligning with the query.
    #     Always Answer in Arabic .\n\n
    #     Conversation history:
    #     {chat_history}


    #     Context:\n {context} \n
    #     Question: \n {question} \n

    #     Answer:
    #     """
    prompt_template = """You are a Biodiversity bot specialized in Marine Species, with a particular focus on those in the Red Sea.  
    Your task is to respond to questions using only the information available in your knowledge base or document repository and always answer in arabic , Be friendly in your responses.  
    **Instructions:**  
    1. **Species Identification:**  
    - Extract the name of the species mentioned in the user's query, if specified.  

    2. **Attribute Matching:**  
    - Identify the specific attribute or information requested in the query (e.g., coordinates, habitat, behavior).  

    3. **Search and Retrieve:**  
    - Search for the extracted species in your documents or vector database.  
    - Locate the attribute relevant to the query for that species.  

    4. **Answer Generation:**  
    - If the requested information is found, provide an accurate and concise response in Arabic.  
    - If the information is not available, inform the user politely that it is not found.  
  

    Conversation history:  
    {chat_history}  

    Context:  
    {context}  

    Question:  
    {question}  

    Answer:  
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", #older version of this file used gemini-pro , gemini pro doesn't #ab
                                temperature=1,
                                top_k=40,
                                top_p=0.95,
                                max_tokens=8192) #ab

    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    
    # query = "Please plot a map with points at latitude 27.321 and longitude 34.123."
    
    # result = agent.run(query)
    # print(f"Success {result}")
    prompt = PromptTemplate(template=prompt_template , input_variable=["chat_history","context" , "question"]) #ab

    # retreival_chain = (
    #     {"context" : retreiver , "question":itemgetter("question")}
    #     | prompt
    #     | llm
    #     | StrOutputParser() 
    # )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question") #ab

    chain = load_qa_chain(llm,chain_type= "stuff" , prompt=llm , memory=memory ) #ab

    return chain
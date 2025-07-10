#importing the libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain #to create a retriever with chat history functionality and create retrieval chain
from langchain.chains.combine_documents import create_stuff_documents_chain #to combine the entire document and send it to the context
from langchain_chroma import Chroma #we are using chroma vectorstore DB
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory #to create conversational chatbot wrt the chat history
from langchain_groq import ChatGroq #we will use groq and Google Gemma 2 model
from langchain_huggingface.embeddings import HuggingFaceEmbeddings #we will use huggingface embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter #t0 create chunks
from langchain_community.document_loaders import PyPDFLoader #we will load pdfs 
import os

#loading the .env
##load_dotenv() 
# for the actual deployment i will do this instead 
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN") --> we do this while running locally
#for a streamlit deployment i willuse this
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]

embeddings = HuggingFaceEmbeddings(model = 'all-MiniLM-L6-v2')

#Setting up streamlit app
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDFs and chat with their content")

#Input the groq api key
api_key = st.text_input("Enter your Groq API Key:", type= "password")

#Check if the api key is provided
if api_key:
    llm = ChatGroq(groq_api_key = api_key, model_name = "Gemma2-9b-It")
    
    #chat interface
    session_id = st.text_input("SessionID", value = "default_session") #giving the session id
    
    #Statefully manage the chat history
    if 'store' not in st.session_state:
        st.session_state.store = {} #in this dictionary we will have all the key value pairs
        
    uploaded_files = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True) #asking user to upload the pdf
    
    #Process these uploaded PDFs
    if uploaded_files:
        documents = [] 
        for uploaded_file in uploaded_files: #to process 1 file at a time incase there are more than 1 pdf uploaded by the user
            #store the file in local memory
            temp_pdf = f"./temp_{uploaded_file.name}"
            with open(temp_pdf, "wb") as file: #opening in writebyte mode
                file.write(uploaded_file.getvalue()) #writing the user uploaded pdf into this temp_pdf that we made
                file_name = uploaded_file.name
                
                #using the loader
                loader = PyPDFLoader(temp_pdf)
                docs = loader.load()
                documents.extend(docs) #these docs are now appended to the documents list we created earlier
                
        #Split(make chunks) and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap =500)
        splits = text_splitter.split_documents(documents=documents) #chunks have been made
        vectorstore = Chroma.from_documents(documents= documents, embedding= embeddings) #storintg these chunks into the vectorstore as vectors
        retriever = vectorstore.as_retriever()
        
        #system prompt
        contextualize_q_system_prompt = (
        """
            Given chat history and the latest user question which might reference context in the chat history,
            formulate a standalone question which can be understood without the chat history.
            Do NOT answer the question, just reformulate it if needed and otherwise return it as it is.
        """
        )
        #prompt template
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"), #we will create chat history soon in the code
                ("human", "{input}"), #user input
            ]
        )
        
        #history aware retriever
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        #Answering the question prompt(Question-answer prompt)
        system_prompt = (
            """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you dont know the answer and the answer is out of the provided context, just say that "I can't find it in this document".
            Use 3-4 sentences at max and keep the answer concise.
            \n\n
            {context}
            """    
        )
        
        #this prompt will be merged with history aware retriever
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        #creating the chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        #creating rag chain  -our history aware retriever which had the contextualize q prompt, is now merged with qa_prompt chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) 
        
        #creating the get session history function
        #we will see the session id and load the chat message history
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store: #if the session id doesnt exist already, we will do this
                st.session_state.store[session_id] = ChatMessageHistory() #all the chat message history for this session will be stored in the session id
            #we created this st.session_state.store empty dictionary above to store all the session ids, now we are storing them
            return st.session_state.store[session_id]
        
        #Creating the conversational rag chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history= get_session_history,
            input_messages_key= "input",
            history_messages_key= "chat_history", #we have seen this earlier in the messagesplaceholder in prompts
            output_messages_key= "answer"
        )
        
        #user input
        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            #calling the conversational rag chain
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={ #for our session id
                    "configurable" : {"session_id": session_id}
                }, #constructs a key 'abc123' in the store
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer']) 
            st.write("Chat History:", session_history.messages)
            
#if the api key is not givn:
else:
    st.warning("Please enter your Groq API key")
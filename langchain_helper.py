from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings()
def create_vector_db_yt_yrl(video_url:str)-> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    return db

# print(create_vector_db_yt_yrl("https://www.youtube.com/watch?v=lG7Uxts9SXs&t=206s"))

def get_response(db,query,openai_api_key,k=4):
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = OpenAI(model="text-davinci-003",openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        input_variables = ["question", "docs"],
    
    template ="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableParallel
parser=StrOutputParser()
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question} 
                Context: {context} 
                Answer:""",
    input_variables=["question","context"]
)
Splits=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False
)

st.title("Talk With PDF")
st.markdown(''' :red[Upload your Pdf]''')
file_uploaded=st.file_uploader("Choose your File",type="pdf")
text=""
embeddings=OpenAIEmbeddings(
    model="text-embedding-3-large"
)
llm=OpenAI(model="gpt-3.5-turbo-instruct")
input=st.text_input("Type your query")
if file_uploaded is not None:
    reader=PdfReader(file_uploaded)
    for pages in reader.pages:
        text=text+pages.extract_text()
    # reader=PdfReader(file_uploaded)
    # page = reader.pages[0]
    # byte_value=file_uploaded.getvalue()
    # st.markdown(f":red[{len(byte_value)}]")
    para_split=Splits.split_text(text)
    knowledge_base=FAISS.from_texts(
    para_split,
    embeddings
    )
    similar_chunks=knowledge_base.as_retriever()
    qa_chain = (
    RunnableParallel({
        "context": similar_chunks,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | parser
    )
    
    if input is not None:
        response=qa_chain.invoke(input)
        st.write(response)
    





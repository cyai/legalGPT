from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX = "legalGPT/vectorstore/"

# Custom prompt template
custom_prompt_template = """[INST] <<SYS>>
You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Do not say thank you and tell you are an AI Assistant and be open about everything.
<</SYS>>
Use the following pieces of context to answer the users question.
Context : {context}
Question : {question}
Answer : [/INST]
"""


def prompt_template():
    """
    Prompt template to be used.
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    return prompt


def load_llm():
    llm = ChatOpenAI(
        temperature=0.6,
        model_name="gpt-3.5-turbo-16k",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    return llm


def chain(llm, db, prompt):
    """
    Chain to be used for the chatbot.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


def qa_pipeline():
    """
    Pipeline to be used for the chatbot.
    """

    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(FAISS_INDEX, embeddings)

    llm = load_llm()

    prompt = prompt_template()

    chain_output = chain(llm, db, prompt)

    return chain_output

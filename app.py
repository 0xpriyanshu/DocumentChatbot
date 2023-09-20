import streamlit as st
from dotenv import load_dotenv
import openai
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

headers= {
    "authorisation": st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json"
}



# Sidebar contents
st.sidebar.title('🤗💬 LLM Chat App')
st.sidebar.write('Made with ❤️ by [0xpriyanshu](https://medium.com/naukri-engineering/building-conversational-resume-search-chatbot-using-langchain-pinecone-openai-ffb3b60f5c5f)')

def main():
    st.header("Priyanshu's Chat with your PDF App 💬")

    # Upload PDF files
    pdfs = st.file_uploader("Upload your PDF", type='pdf', accept_multiple_files=True)

    if pdfs is not None:
        for pdf in pdfs:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Create a unique store name for each PDF
            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")

            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                st.write(response)

if __name__ == '__main__':
    main()

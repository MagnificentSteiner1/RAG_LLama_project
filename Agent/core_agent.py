from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

db_location = Path(__file__).resolve().parent.parent / "Database"

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

model = OllamaLLM(
    model="llama3.2"
)

vector_store = Chroma(
    collection_name="CORD-19",
    persist_directory=db_location,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)
template = """
You need to give information and advice for a certain given question relating to Covid 19. Only use the information given from the context. 
If you dont have sufficient information from the context, dont fill the gaps with anything outside of the context, and respond that you are not able to assist with the given question.

This is the topic in question : {question}
And here is the relevant dataset : {context}
"""

if __name__ == "__main__":
    while True:
        question = input("What is your question? Press q to quit.")
        if question == "q":
            break
        top_k_dataset = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in top_k_dataset])

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        result = chain.invoke({"question": question,"context": context})
        print(result)
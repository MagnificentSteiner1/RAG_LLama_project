from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from doc_embed import retriever

model = OllamaLLM(model="llama3.2")
template = """
Your need to give advice on which country would be the best one for me to live in, 
based on my life situation and preferences and the quality of life dataset that will be provided.

This is my life situation and preferences : {preferences}
And here is the relevant dataset : {dataset}
"""
while True:
    question = input("What is your situation? Press q to quit.")
    if question == "q":
        break
    situation = retriever.invoke(question)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    result = chain.invoke({"preferences":question, "dataset":situation})
    print(result)
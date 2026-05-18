from langchain_core.prompts import ChatPromptTemplate
from Agent.core_agent import retriever, model

template = """
You need to give information and advice for a certain given question relating to Covid 19. Only use the information given from the context. 
If you dont have sufficient information from the context, dont fill the gaps with anything outside of the context, and respond that you are not able to assist with the given question.

This is the topic in question : {question}
And here is the relevant dataset : {context}

Synthesize information across multiple context chunks when relevant.
Prioritize consensus across sources.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

def AskQuestion(question: str):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    result = chain.invoke({
        "question": question,
        "context": context
    })
    print(question)
    print("TOP DOC SAMPLE:")
    for i, doc in enumerate(retrieved_docs[:3]):
        print(i, doc.page_content[:300])
    return {
        "answer":result,
        "context":[doc.page_content for doc in retrieved_docs]
    }
import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
import chromadb.utils.embedding_functions as embedding_functions
import streamlit as st
import query_chroma_db as ui
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main(prompt):
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    query_text = prompt

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(api_key='API_KEY' )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=8)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    model = ChatOpenAI(api_key='API_KEYxh6QP1')
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = (f"\n\nWARNING: The below content is confidential and intended for "
                          f"BAT specified recipient only."
                          f" It is strictly forbidden to share any part of this message with any "
                          f"third party, without the consent of the owner.\n\n "
                          f"\n{response_text}\n\n I hope I was able to answer the question"
                          f"\n\nFor any queries related to DOC GENIE please contact ANALYTICAL WIZARDS"
                        )
    print(formatted_response)
    return formatted_response


if __name__ == "__main__":
    main()

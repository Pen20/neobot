import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Create the Neo4jVector retriever
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="student_embeddding_index",
    node_label="Student",
    text_node_property="llm_response",
    embedding_node_property="student_error_embedding",
    retrieval_query="""
RETURN
    node.llm_response AS text,
    score,
    {
        student_id: node.student_id,
        grade: [(node)-[:ANSWERED]->(a: Answer)-[:FOR_QUESTION]->(q: Question) | [node.student_id, a.grade]],
        response: [(node)-[:ANSWERED]->(a: Answer)-[:FOR_QUESTION]->(q: Question) | [node.student_id, a.response]],
        right_answer: [(node)-[:ANSWERED]->(a: Answer)-[:FOR_QUESTION]->(q: Question) | [node.student_id, a.right_answer]],
        question_id: [(node)-[:ANSWERED]->(a: Answer)-[:FOR_QUESTION]->(q: Question) | [node.student_id, q.question_id]]
    } AS metadata
"""
)


# Create the retriever
retriever = neo4jvector.as_retriever()

# Prompt for the vector QA chain
instructions = (
    "Use the given context and the information in medata to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}"),
])

# Build the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
error_retriever = create_retrieval_chain(retriever, question_answer_chain)

# General error feedback retrieval
def get_student_error_feedback(input):
    return error_retriever.invoke({"input": input})


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
    index_name="student_embedding_index",
    node_label="Student",
    text_node_property="llm_response",
    embedding_node_property="student_error_embedding",
    retrieval_query="""
RETURN node.llm_response AS text, score,
{
  studentId: node.student_id,
  grade: [(node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | Answer.grade],
  studentResponse: [(node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | Answer.response],
  rightSolution: [(node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | Answer.right_answer],
  questionID: [(node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | Question.question_id]
} AS metadata

"""
)


# Create the retriever
retriever = neo4jvector.as_retriever()

# Prompt for the vector QA chain
instructions = (
    "You are an educational assistant helping to analyze student responses. "
    "The provided context relates to each student's performance and the errors they made while answering questions. "
    "If there is no information about a student or a specific question, it means the student answered it correctly and was therefore excluded from the error data. "
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


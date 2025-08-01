import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Create the Neo4jVector retriever
neo4jvector_category = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="category_embedding_index",
    node_label="ErrorCategory",
    text_node_property="error_category",
    embedding_node_property="error_category_embedding",
    retrieval_query="""
MATCH (node:ErrorCategory)<-[:TAGGED_WITH]-(seed:Seed)<-[:USED_SEED]-(q:Question)<-[:FOR_QUESTION]-(a:Answer)<-[:ANSWERED]-(s:Student)
WHERE s.student_id = $student_id  // Only run this part if variable exists
RETURN substring(node.error_category, 0, 300) AS text, score,
{
  studentId: s.student_id,
  studentResponse: substring(a.response, 0, 300),
  questionID: q.question_id
} AS metadata

"""
)

# Create the retriever
retriever_category = neo4jvector_category.as_retriever(search_kwargs={"k": 1})


# Prompt for the vector QA chain
instructions = (
    "You are an educational assistant helping to analyze student responses. "
    "The provided context relates to each student's error categories, as classified according to Newman's Error Analysis (NEA) theory. "
    "If there is no information about a student or a specific question, it means the student answered it correctly and was therefore excluded from the error data. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

prompt_category = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}"),
])

# Build the chain
question_answer_chain_category = create_stuff_documents_chain(llm, prompt_category)
nea_retriever = create_retrieval_chain(retriever_category, question_answer_chain_category)

# General error feedback retrieval function
def get_student_nea_category_feedback(input):
    return nea_retriever.invoke({"input": input})

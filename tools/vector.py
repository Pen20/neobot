import streamlit as st
from llm import llm, embeddings
from graph import graph

# tag::import_vector[]
from langchain_neo4j import Neo4jVector
# end::import_vector[]
# tag::import_chain[]
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# end::import_chain[]

# tag::import_chat_prompt[]
from langchain_core.prompts import ChatPromptTemplate
# end::import_chat_prompt[]

# Create the Neo4jVector

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="student_embeddding_index",                 # (3)
    node_label="Student",                      # (4)
    text_node_property="llm_response",               # (5)
    embedding_node_property="student_error_embedding", # (6)
    retrieval_query="""
RETURN
    node.llm_response AS text,
    score,
    {
        student_id: node.student_id,
        grade: [ (node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [node.student_id, Answer.grade] ],
        student_response: [ (node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [node.student_id, Answer.response] ],
        original_answer: [ (node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [node.student_id, Answer.right_answer] ],
        question_id: [ (node)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [node.student_id, Question.question_id] ]
    } AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
error_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)


# Create a function to call the chain
def get_student_error_feedback(input):
    return error_retriever.invoke({"input": input})
import streamlit as st
from llm import llm
from graph import graph

# Create the Cypher QA chain
from langchain_neo4j import GraphCypherQAChain

# Fine-tuning
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about mathematics and student performance.

Convert the question into a Cypher query using the provided schema. Do not invent properties. Use the following format.


Example:

What is the grade of the student 263 on question 2?
```
MATCH  (s:Student {student_id: "263"})-[:ANSWERED]->(a: Answer)-[:FOR_QUESTION]->(q: Question {question_id: "Q. 2"})
RETURN s.student_id, a.grade, q.question_id
```

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)
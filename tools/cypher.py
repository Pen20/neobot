import streamlit as st
from llm import llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain

# Create the Cypher QA chain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about mathematics and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Use the following format for your Cypher queries:
Example Cypher Statements:

1. What mistake did student 12 make on question 7?
```
MATCH (s:Student {student_id: "12"})-[:ANSWERED]->(a:Answer)-[:FOR_QUESTION]->(q:Question {question_id: "7"})
RETURN a.llm_response AS mistake, a.response AS student_answer, a.right_answer AS correct_answer

```
2. Which misconceptions did student 23 have in their answer to question 4?
```
MATCH (s:Student {student_id: "23"})-[:ANSWERED]->(a:Answer)-[:FOR_QUESTION]->(q:Question {question_id: "4"})
RETURN a.llm_response AS explanation

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
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)
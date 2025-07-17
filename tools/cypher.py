import streamlit as st
from llm import llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain

# Create the Cypher QA chain
from langchain.prompts.prompt import PromptTemplate

# 🔐 Get OpenAI token from Streamlit secrets
openai_token = st.secrets["OPENAI_API_KEY"]

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about mathematics and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Use the following format for your Cypher queries:
Example Cypher Statements:

1. What mistake did student 12 make on question 5?
```
MATCH (s:Student {student_id: "12"})-[:ANSWERED]->(a:Answer)-[:FOR_QUESTION]->(q:Question {question_id: "Q. 5"})-[:USED_SEED]->(sd:Seed)-[:TAGGED_WITH]->(ec:ErrorCategory)
RETURN
    s.llm_response AS text,
    {
        student_id: s.student_id,
        grade: [ (s)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [s.student_id, Answer.grade] ],
        student_response: [ (s)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [s.student_id, Answer.response] ],
        original_answer: [ (s)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [s.student_id, Answer.right_answer] ],
        question_id: q.question_id
    } AS metadata

```
2. Which misconceptions did student 23 have in their answer to question 1?
```
MATCH (s:Student {student_id: "23"})-[:ANSWERED]->(a:Answer)-[:FOR_QUESTION]->(q:Question {question_id: "Q. 1"})-[:USED_SEED]->(sd:Seed)-[:TAGGED_WITH]->(ec:ErrorCategory)
RETURN
    s.llm_response AS mistakes, 
    ec.error_category AS categories,
    {
        student_id: s.student_id,
        grade: [ (s)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [s.student_id, Answer.grade] ],
        student_response: [ (s)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [s.student_id, Answer.response] ],
        original_answer: [ (s)-[:ANSWERED]->(Answer)-[:FOR_QUESTION]->(Question) | [s.student_id, Answer.right_answer] ],
        question_id: q.question_id
    } AS metadata
```

3. Which similar error categories appeared in students’ answers to question 5?
```
WWITH genai.vector.encode(
"Which similar error categories appeared in students’ answers to question 5?",
"OpenAI",
{{ token: "{openai_token}" }}
) AS categoryEmbedding

CALL db.index.vector.queryNodes('category_embedding_index', 50, categoryEmbedding)
YIELD node AS similarCategory, score

MATCH (s:Student)-[:ANSWERED]->(a:Answer)-[:FOR_QUESTION]->(q:Question {{question_id: "Q. 5"}})
-[:USED_SEED]->(:Seed)-[:TAGGED_WITH]->(ec:ErrorCategory)
WHERE ec = similarCategory

RETURN
ec.error_category AS category,
score,
s.student_id AS student_id,
a.grade AS grade,
a.response AS student_response,
a.right_answer AS original_answer,
q.question_id AS question_id
ORDER BY score DESC
LIMIT 10
```
4. Which 10 top-performing students made transformation errors on question 1?
```
MATCH (s:Student)-[:ANSWERED]->(a:Answer)-[:FOR_QUESTION]->(q:Question {question_id: "Q. 1"})-[:CATEGORIZED_AS]->(nec:ErrorCategory)
WHERE nec.error_category CONTAINS "transformation"
  AND a.grade >= 0.6
RETURN DISTINCT s.student_id, a.grade, nec.error_category
ORDER BY a.grade DESC
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
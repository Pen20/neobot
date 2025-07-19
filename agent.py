from llm import llm
from graph import graph

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

from tools.vector import get_student_error_feedback
from utils import get_session_id

from tools.cypher import cypher_qa


# --------------------------
# Custom tool wrapper
# --------------------------

def formatted_student_error_feedback(input):
    result = get_student_error_feedback(input)
    answer = result.get("answer", "")
    docs = result.get("context", [])

    response = f"Explanation:\n{answer}\n"

    if docs:
        doc = docs[0]
        metadata = doc.metadata

        student_id = metadata.get("student_id", "Unknown")
        grade = metadata.get("grade", [["", ""]])[0][1] if metadata.get("grade") else "N/A"
        question_id = metadata.get("question_id", [["", ""]])[0][1] if metadata.get("question_id") else "N/A"

        response += f"\nStudent ID: {student_id}"
        response += f"\nQuestion ID: {question_id}"
        response += f"\nGrade: {grade}"
    else:
        response += "\nNo metadata found."

    return response

# --------------------------
# Prompts and chains
# --------------------------

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a tutor expert providing information about mathematics."),
        ("human", "{input}"),
    ]
)

math_chat = chat_prompt | llm | StrOutputParser()

# --------------------------
# Tools
# --------------------------

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general mathematics chat not covered by other tools.",
        func=math_chat.invoke,
    ),
    Tool.from_function(
        name="Student Error Feedback",
        description="Analyze and provide feedback on student errors using LLM explanations and data retrieved from the database.",
        func=formatted_student_error_feedback,
    ),
    Tool.from_function(
        name="Math QA information",
        description="Provide information on student about questions using Cypher",
        func = cypher_qa
    )
]

# --------------------------
# Memory & Agent Setup
# --------------------------

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a tutor expert providing information about mathematics.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to mathematics.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

# Agent and executor
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Handler to call the Conversational agent and return a response for the UI.
    """
    try:
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}},
        )
        return response['output']
    except Exception as e:
        return f"Sorry, an error occurred: {e}"
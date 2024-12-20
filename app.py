import streamlit as st
import os
import json
import random
import sqlite3
from typing import Dict, Any
from dotenv import load_dotenv

from crewai import Agent, Task, Crew
from crewai_tools import Tool

load_dotenv()  # load environment variables (e.g., OPENAI_API_KEY)

st.set_page_config(page_title="Web Weavers", page_icon=":spider_web:")

st.title("Web Weavers")
st.subheader("Agents to weave knowledge and reasoning into LLMs")

# ----------------------------
# Setup: Reading Databases and Schema
# ----------------------------
BASE_DIR = "./data/database/"
db_map = {}

for db_id in os.listdir(BASE_DIR):
    db_dir_path = os.path.join(BASE_DIR, db_id)
    if os.path.isdir(db_dir_path):
        sqlite_path = os.path.join(db_dir_path, f"{db_id}.sqlite")
        db_path = os.path.join(db_dir_path, f"{db_id}.db")
        if os.path.exists(sqlite_path):
            db_map[db_id] = sqlite_path
        elif os.path.exists(db_path):
            db_map[db_id] = db_path

schema_path = "./data/schema_info.json"
with open(schema_path, 'r') as f:
    schema_info = json.load(f)

class SchemaInfoTool(Tool):
    def __init__(self):
        super().__init__(
            name="schema_info_tool",
            description="Return the schema_info for a given db_id, along with question and reasoning_type.",
            func=self.run
        )

    def run(self, question: str, db_id: str, reasoning_type: str) -> Dict[str, Any]:
        db_schema = schema_info.get(db_id, {})
        return {
            "schema_info": db_schema,
            "db_id": db_id,
            "reasoning_type": reasoning_type,
            "question": question
        }

class SQLExecutionTool(Tool):
    def __init__(self):
        super().__init__(
            name="sql_execution_tool",
            description="Execute the provided SQL query on the specified db_id and return the results.",
            func=self.run
        )

    def run(self, sql_query: str, db_id: str) -> Dict[str, Any]:
        if db_id not in db_map:
            return {"error": f"No database found for db_id={db_id}", "sql_query": sql_query, "db_id": db_id}

        db_path = db_map[db_id]
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        results = []
        columns = []
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        except Exception as e:
            print(f"Error executing query: {e}")
        finally:
            conn.close()

        return {
            "columns": columns,
            "rows": results,
            "db_id": db_id,
            "sql_query": sql_query
        }

knowledge_agent = Agent(
    name="knowledge_agent",
    role="System",
    goal="Provide schema information and other database knowledge.",
    backstory="This agent holds database schema details.",
    tools=[SchemaInfoTool()],
    llm="gpt-4o-mini"
)

knowledge_task = Task(
    description=(
        "You are a system agent. The user provides a question, db_id, and reasoning_type. "
        "Use the schema_info_tool to retrieve the schema information for the given db_id."
    ),
    expected_output="Schema info and context",
    agent=knowledge_agent,
    parameters={
        "question": "{question}",
        "db_id": "{db_id}",
        "reasoning_type": "{reasoning_type}"
    }
)

schema_linking_agent = Agent(
    name="schema_linking_agent",
    role="System",
    goal="Link user query to database schema elements.",
    backstory="This agent uses the schema info to find relevant tables/columns."
)

schema_linking_task = Task(
    description=(
        "Given the schema_info and the user question, identify relevant tables and columns. "
        "Return linked_tables and linked_columns as JSON."
    ),
    expected_output="Linked schema elements",
    agent=schema_linking_agent,
    parameters={
        "schema_info": knowledge_task,
        "question": "{question}",
        "db_id": "{db_id}",
        "reasoning_type": "{reasoning_type}"
    }
)

sql_generation_agent = Agent(
    name="sql_generation_agent",
    role="System",
    goal="Generate a correct SQL query from the question and schema links.",
    backstory="This agent uses reasoning to produce an SQL query.",
    llm="gpt-4o-mini"
)

sql_generation_task = Task(
    description=(
        "Given the linked_tables and linked_columns, and the question, generate a valid SQL query."
    ),
    expected_output="SQL query string",
    agent=sql_generation_agent,
    parameters={
        "linked_tables": schema_linking_task,
        "linked_columns": schema_linking_task,
        "question": "{question}",
        "db_id": "{db_id}",
        "reasoning_type": "{reasoning_type}"
    }
)

sql_execution_agent = Agent(
    name="sql_execution_agent",
    role="System",
    goal="Execute the SQL query and return the results.",
    backstory="This agent runs the SQL against the specified database.",
    tools=[SQLExecutionTool()]
)

sql_execution_task = Task(
    description=(
        "Use the sql_execution_tool to run the given SQL query on db_id and return the results."
    ),
    expected_output="Query results",
    agent=sql_execution_agent,
    parameters={
        "sql_query": sql_generation_task,
        "db_id": "{db_id}",
        "reasoning_type": "{reasoning_type}",
        "question": "{question}"
    }
)

crew = Crew(
    agents=[
        knowledge_agent,
        schema_linking_agent,
        sql_generation_agent,
        sql_execution_agent
    ],
    tasks=[
        knowledge_task,
        schema_linking_task,
        sql_generation_task,
        sql_execution_task
    ]
)

dev_path = "./data/en_data/dev.json"
with open(dev_path, 'r') as f:
    dev_data = json.load(f)

st.write("**Below is a randomly selected question from the Archer dev set.**")

if st.button("Pick Random Question"):
    sample = random.choice(dev_data)
    question = sample["question"]
    selected_db = sample["db_id"]
    reasoning_type = sample.get("reasoning_type", "- - -")

    st.write(f"**Question:** {question}")
    st.write(f"**Database ID:** {selected_db}")
    st.write(f"**Reasoning Type:** {reasoning_type}")

    # Run the entire pipeline using crew.kickoff
    result = crew.kickoff(
        inputs={
            "question": question,
            "db_id": selected_db,
            "reasoning_type": reasoning_type
        }
    )

    # Display final result
    st.markdown("### Final Agentic Result")
    st.write(result)

    # If result contains intermediate results, display them.
    # For example, if result is a dict that includes intermediate steps:
    # This depends on how crew.kickoff() structures its output.
    # If available:
    if "steps" in result:
        st.markdown("### Intermediate Steps")
        for step_name, step_output in result["steps"].items():
            st.markdown(f"**{step_name}**")
            st.write(step_output)

    # Vanilla LLM approach: Just prompt the sql_generation_agent with minimal info
    # Without schema linking:
    vanilla_sql_res = sql_generation_agent.tools[0].func(question=question, db_id=selected_db, reasoning_type=reasoning_type) if sql_generation_agent.tools else {}
    # If no tools are involved, you may try a different approach to get a baseline.
    # Or skip the vanilla approach if no direct method is available.
    st.markdown("### Vanilla LLM (No Knowledge / No Schema Linking)")
    if "sql_query" in vanilla_sql_res:
        vanilla_exec_res = sql_execution_agent.tools[0].func(sql_query=vanilla_sql_res["sql_query"], db_id=selected_db)
        st.write("Vanilla LLM Execution Result:")
        st.write(vanilla_exec_res)
    else:
        st.write("Vanilla LLM did not produce a valid SQL query or no direct baseline method available.")

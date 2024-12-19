# Archer Text-to-SQL with CrewAI

This repository contains an example setup for running a text-to-SQL pipeline on the Archer benchmark dataset using CrewAI agents and tasks.

## Overview

We have implemented a series of agents and tasks using CrewAI that:

1. Load and provide database schema information (`knowledge_agent`).
2. Link user queries to relevant database tables and columns (`schema_linking_agent`).
3. Generate SQL queries from the user’s natural language question (`sql_generation_agent`).
4. Execute the generated SQL queries on the corresponding SQLite database (`sql_execution_agent`).

Additionally, we have:

- Automatically generated a `schema_info.json` file by introspecting the SQLite databases.
- Created a `db_map` dynamically by scanning the directories containing databases.

## Requirements

### System Dependencies

- **SQLite**:  
  If you are using Fedora, CentOS, or RHEL-based systems with `dnf`, you can install SQLite via:
  ```bash
  sudo dnf install sqlite

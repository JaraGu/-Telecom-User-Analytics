# Data Folder README

This folder contains data for the project in the form of PostgreSQL dump files.

## Files

- `telecom.sql`: The main data dump file containing project-related data.
- `schema_dump.sql`: The schema dump file providing the database structure.

## Loading Data

To load the data into a PostgreSQL database, follow these steps:

### Prerequisites

- Ensure you have PostgreSQL installed on your machine or server.

### Create an Empty Database

Create an empty PostgreSQL database for restoring the dump data:

CREATE DATABASE your_database_name;

Replace <your_database_name> with the desired name for your database.

### Loading Schema

Run the following command to load the schema into your PostgreSQL database:

psql -U your_username -h localhost -d tyour_database_name -f /path/to/schema.sql

### Loading Data

Run the following command to load the main data into your PostgreSQL database:

psql -U your_username -h localhost -d tyour_database_name -f /path/to/telecom.sql

### Additional Notes

- Make sure your PostgreSQL server is running before running the commands.
- Adjust the connection details in the commands based on your PostgreSQL setup.

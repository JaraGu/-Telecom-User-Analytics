import pandas as pd
from sqlalchemy import create_engine
import os


def load_data(credentials_file='env_vars.txt', database_name='telecom', table_name='xdr_data'):
    # Get the current directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the full path to the credentials file
    credentials_file_path = os.path.join(script_dir, '..', credentials_file)

    # Read database credentials from file
    user, password, host, port = read_db_credentials(credentials_file_path)

    # Create the database engine
    connection_params = {"host": host, "user": user,
                         "password": password, "port": port, "database": database_name}
    engine = create_engine(
        f"postgresql+psycopg2://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")

    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'

    # Read data into a DataFrame
    df = pd.read_sql(sql_query, con=engine)

    return df


def write_to_sql(df, table_name, credentials_file='env_vars.txt', database_name='telecom'):
    # Get the current directory of the script
    script_dir = os.path.dirname(__file__)

    # Build the full path to the credentials file
    credentials_file_path = os.path.join(script_dir, '..', credentials_file)

    # Read database credentials from file
    user, password, host, port = read_db_credentials(credentials_file_path)

    # Create the database engine
    connection_params = {"host": host, "user": user,
                         "password": password, "port": port, "database": database_name}
    engine = create_engine(
        f"postgresql+psycopg2://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")

    # Write data to a new table
    df.to_sql(name=table_name, con=engine, index=False, if_exists='replace')


def read_db_credentials(credentials_file):
    # Read database credentials from the specified file
    with open(credentials_file, 'r') as file:
        lines = file.readlines()

    # Extract individual credential values
    user = lines[0].strip().split('=')[1]
    password = lines[1].strip().split('=')[1]
    host = lines[2].strip().split('=')[1]
    port = lines[3].strip().split('=')[1]

    return user, password, host, port

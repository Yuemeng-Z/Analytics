import pandas as pd
import pyodbc

from pathlib import Path


def render_sql_file(sql_file_path, replacements=None):
  """
  Read a SQL template from a txt file and replace placeholder tokens.

  Example:
      query = render_sql_file(
          "Project Sunrise/sqls/query_everD.txt",
          {"TARGET_MOC": 12, "EVER_D_DAYS": 60}
      )
  """
  query = Path(sql_file_path).read_text(encoding="utf-8")

  if not replacements:
    return query

  for placeholder, value in replacements.items():
    query = query.replace(str(placeholder), str(value))

  return query


def connect_db_with_inputs(sql_file_path, replacements=None):
  """
  Read a SQL txt file, replace placeholders, query the database, and return a dataframe.

  Example:
      df = connect_db_with_inputs(
          "Project Sunrise/sqls/query_everD.txt",
          {"TARGET_MOC": 12, "EVER_D_DAYS": 60}
      )
  """
  query = render_sql_file(sql_file_path, replacements)
  return connect_db(query)


def connect_db(query):
  # Create connection
  conn = pyodbc.connect('Driver={SQL Server};'
                        'Server=CL03N02\LP02;'
                        'Database=ConsumerLoans;'
                        'Trusted_Connection=yes;'
                        )

  # Load into pandas dataframe
  df = pd.read_sql(query, conn)

  # Close connection
  conn.close()
  
  return df


def connect_db_temp_table_with_query(temp_table_sql, table_creation_queries_dict):
  # Create connection
  conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=CL03N02\LP02;'
                      'Database=ConsumerLoans;'
                      'Trusted_Connection=yes;'
                      )

  # 1. Run setup SQL that creates #cleanTable
  cursor = conn.cursor()
  cursor.execute(temp_table_sql)
  cursor.commit()

  # 2. Run each output query separately, using the same conn
  res_dict = {}
  for table_name, table_creation_query in table_creation_queries_dict.items():
      res_df = pd.read_sql(table_creation_query, conn)
      res_dict[table_name] = res_df

  conn.close()
  return res_dict

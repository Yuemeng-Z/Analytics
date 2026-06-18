import pandas as pd
import pyodbc

def connect_db(query):
  # Create connection
  conn = pyodbc.connect('Driver={SQL Server};'
                        'Server=CL03N02\LP02;'
                        'Database=ConsumerLoans;'
                        'Trusted_Connection=yes;'
                      #   'UID=BFTG-NT\YuemengZhang;'
                      #   'PWD=@WSX3edc$RFV'
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
                    #   'UID=BFTG-NT\YuemengZhang;'
                    #   'PWD=@WSX3edc$RFV'
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
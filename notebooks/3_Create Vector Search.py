# Databricks notebook source
# DBTITLE 1,Install Databricks SDK and Vector Search Library
# MAGIC %pip install -U --quiet databricks-sdk==0.28.0 databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Config File
catalogName = "mzervou"
dbName = "media"
VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-16"
embedding_model_endpoint_name = 'databricks-gte-large-en'
print(catalogName)
print(dbName)

# COMMAND ----------

# DBTITLE 1,Query Cleaned Data
sql_query = f"""
SELECT movie_id FROM `{catalogName}`.`{dbName}`.`generated_reviews` group by movie_id
"""
display(spark.sql(sql_query))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time
import helper_functions as help

# Initialize the Vector Search Client with the option to disable the notice.
vsc = VectorSearchClient(disable_notice=True)


# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed for Conformed Table
# This allows us to create a vector search index on the table
tables_tb_indexed = ["generated_reviews", "emails_reports"]

for table_name in tables_tb_indexed:
  sql_query = f"""
  ALTER TABLE `{catalogName}`.`{dbName}`.`{table_name}` 
  SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
  """
  spark.sql(sql_query)

  vs_index = f"{catalogName}.{dbName}.{table_name}_index"
  source_table = f"{catalogName}.{dbName}.{table_name}"

  if table_name == "generated_reviews":
    primary_key = 'movie_id'
    embedding_source_column = 'generated_review'
  else:
    primary_key = 'Content'
    embedding_source_column = '_3'

  print(f"Creating index {vs_index} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")

  # Create a new delta sync index on the vector search endpoint.
  # This index is created from a source Delta table and is kept in sync with the source table.
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,  # The name of the vector search endpoint.
    index_name=vs_index,  # The name of the index to create.
    source_table_name=source_table,  # The full name of the source Delta table.
    pipeline_type="TRIGGERED",  # The type of pipeline to keep the index in sync with the source table.
    primary_key=primary_key,  # The primary key column of the source table.
    embedding_source_column=embedding_source_column,  # The column to use for generating embeddings.
    embedding_model_endpoint_name=embedding_model_endpoint_name  # The name of the embedding model endpoint.
  )

  # Wait for the index to be fully operational before proceeding.
  help.wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index)

  # Print a confirmation message indicating the index is ready for use.
  print(f"index {vs_index} on table {source_table} is ready")

# COMMAND ----------

# DBTITLE 1,Demonstration of Similarity Search Using Vector Search Index
# Similarity Search

# Define the query text for the similarity search.
query_text = "What are the comments about dJANGO"

# Perform a similarity search on the vector search index.
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=query_text,
  columns=['generated_review', 'movie_id'],
  num_results=5)  # Specify the number of results to return.

results

# COMMAND ----------



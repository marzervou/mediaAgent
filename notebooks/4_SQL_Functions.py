# Databricks notebook source
catalogName = "mzervou"
dbName = "media"

print(catalogName)
print(dbName)

# COMMAND ----------

# MAGIC %md
# MAGIC Average Watch Time per Genre

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   cm.broadcaster as broadcaster, 
# MAGIC   AVG(vs.watch_time) AS avg_watch_time
# MAGIC FROM mzervou.media.viewing_statistics vs
# MAGIC JOIN shared.chris_williams.tvdemo_gold_tv cm 
# MAGIC ON vs.content_id = cm.showing_PK
# MAGIC GROUP BY cm.broadcaster

# COMMAND ----------

broadcaster = "BBC"

spark.sql(f'''
CREATE OR REPLACE FUNCTION {catalogName}.{dbName}.avg_watch_per_broadcaster(
    input_broadcaster STRING COMMENT "the broadcaster to calculte average watch time for" DEFAULT "{broadcaster}"
)
RETURNS TABLE (
    broadcaster STRING,
    avg_watch_time DOUBLE
)
COMMENT "This function returns the average watch time across all movies for a given broadcaster"
RETURN
SELECT 
  cm.broadcaster , 
  AVG(vs.watch_time) AS avg_watch_time
FROM mzervou.media.viewing_statistics vs
JOIN shared.chris_williams.tvdemo_gold_tv cm 
ON vs.content_id = cm.showing_PK
WHERE  cm.broadcaster = input_broadcaster
GROUP BY cm.broadcaster
''')


# COMMAND ----------

spark.sql(f"SELECT * FROM {catalogName}.{dbName}.avg_watch_per_broadcaster('ITV')").display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We define a function that takes a product as input and returns all upstream relationships (product → raw).

# COMMAND ----------

spark.sql(f'''
CREATE OR REPLACE FUNCTION {catalogName}.{dbName}.content_comment_count(
    input_broadcaster STRING COMMENT "the broadcaster to calculate comment counts for"
)
RETURNS TABLE (
    content_id STRING,
    title STRING,
    broadcaster STRING,
    comment_count BIGINT,
    percentage_of_total DOUBLE
)
COMMENT "Returns the comment count and percentage for content items across all movies the given broadcaster showed, filtered by broadcaster, compared to all comments"
RETURN
WITH comment_counts AS (
    SELECT
      ui.content_id,
      tv.title,
      tv.broadcaster,
      COUNT(*) AS comment_count
    FROM
      mzervou.media.user_interactions ui
      JOIN shared.chris_williams.tvdemo_gold_tv tv ON ui.content_id = tv.showing_PK
    WHERE
      ui.action ILIKE '%comment%'
    GROUP BY
      ui.content_id,
      tv.title,
      tv.broadcaster
),
total_comments AS (
    SELECT SUM(comment_count) AS total
    FROM comment_counts
)
SELECT
  cc.content_id,
  cc.title,
  cc.broadcaster,
  cc.comment_count,
  (cc.comment_count * 100.0 / tc.total) AS percentage_of_total
FROM
  comment_counts cc,
  total_comments tc
WHERE
  cc.broadcaster = input_broadcaster
ORDER BY
  cc.comment_count DESC
''')


# COMMAND ----------

spark.sql(f"SELECT * FROM {catalogName}.{dbName}.content_comment_count('ITV')").display()

# COMMAND ----------



# Databricks notebook source
catalogName = "mzervou"
dbName = "media"

print(catalogName)
print(dbName)

# COMMAND ----------

from datetime import datetime, timedelta
import random

# Define the date range
start_date = datetime.fromisoformat("2023-11-28T00:00:00.000+00:00")
end_date = datetime.fromisoformat("2024-11-25T00:00:00.000+00:00")

# Function to generate random dates
def generate_random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# List of genres
genres = ["Sci-fi", "Drama", "Comedy", "Documentary", "Action"]

# Generate email examples for each genre
email_examples_random = []
for genre in genres:
    # Generate random parameters
    random_date = generate_random_date(start_date, end_date).isoformat()
    avg_watch_time = random.randint(30, 120)
    genre_avg_watch_time = random.randint(20, 100)
    new_subscriptions = random.randint(50, 500)
    retention_increase = round(random.uniform(1.0, 10.0), 1)
    engagement_percentage = random.randint(20, 80)
    weeks = random.randint(1, 8)
    target_demographic = random.choice(["18-24", "25-34", "35-44", "45-54"])
    lead_actor = random.choice(["Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Scarlett Johansson"])
    positive_mentions = random.randint(10, 100)
    sub_genre = random.choice(["Space Opera", "Romantic Comedy", "True Crime", "Superhero", "Historical"])
    target_age_group = random.choice(["18-24", "25-34", "35-44", "45-54"])
    platform = random.choice(["Mobile", "Web", "Smart TV"])
    weakness = random.choice(["Pacing", "Character Development", "Visual Effects", "Plot Complexity"])
    analyst_name = random.choice(["John Smith", "Jane Doe", "Alex Johnson", "Sarah Lee"])

    # Create email content
    email_content = f"""
Date: {random_date}  

Genre Analysis Report for {genre}

Dear Content Team,

We've analyzed the performance and feedback for the {genre} genre. Key insights include:

1. **Average Watch Time:** Users are watching {avg_watch_time} minutes of {genre} content, compared to the overall average of {genre_avg_watch_time} minutes.
2. **Impact:** {genre} content has led to {new_subscriptions} new subscriptions and {retention_increase}% increase in user retention.
3. **Engagement:** {engagement_percentage}% of viewers have interacted (liked, shared, or commented) with {genre} content in the last {weeks} weeks.

**Content Highlights:**
- **Demographic Appeal:** Particularly popular among the {target_demographic} age group.
- **Performer Impact:** {lead_actor} received {positive_mentions} positive mentions in {genre} content reviews.

**Recommendations:**
- **Similar Content:** Consider producing more content in the {sub_genre} sub-genre of {genre}.
- **Promotion Strategy:** Target marketing towards the {target_age_group} age group on the {platform} platform for {genre} content.
- **Improvement Areas:** Address {weakness} in future {genre} productions to enhance viewer satisfaction.

We'll review these insights in our next content strategy meeting to align on future directions for {genre} content.

Best regards,
{analyst_name}
Genre Performance Analyst
"""
    email_examples_random.append((random_date, genre, email_content.strip()))


print("Genre analysis reports have been generated and saved to 'genre_analysis_reports.csv'")

df_emails_random = spark.createDataFrame(email_examples_random, schema=["Date", "Content"])

# Save to a table
df_emails_random.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalogName}.{dbName}.emails_reports")


# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `mzervou`.`media`.`emails_reports` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

df_emails_random_spark = spark.read.table(f"{catalogName}.{dbName}.emails_reports")
display(df_emails_random_spark.limit(10))

# COMMAND ----------

content_df = spark.sql(f'''SELECT 
    user_id,
    showing_PK AS movie_id,
    title,
    description as description
FROM
    mzervou.media.viewing_statistics INNER JOIN shared.chris_williams.tvdemo_gold_tv  
    ON content_id = showing_PK   
GROUP BY ALL
ORDER BY RAND()
limit 1000''')

# COMMAND ----------

content_df.display()

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

# Define parameters
endpoint_name = "mz-llama3"
prompt = "Generate a movie review for the content with this description: "
input_column_name = "description"
num_output_tokens = 300
temperature = 0.7
output_column_name = "generated_review"
output_table_name = f"{catalogName}.{dbName}.generated_reviews"

# Generate reviews using the LLM and include the movie ID
df_out = content_df.selectExpr(
    "movie_id",
    f"ai_query('{endpoint_name}', CONCAT('{prompt}', {input_column_name}), modelParameters => named_struct('max_tokens', {num_output_tokens},'temperature', {temperature})) as {output_column_name}"
)

# Write the results to a Delta table
df_out.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(output_table_name)

print(f"Generated reviews have been saved to the table: {output_table_name}")


# COMMAND ----------

df_out.display()

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `mzervou`.`media`.`generated_reviews` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------



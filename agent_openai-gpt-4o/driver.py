# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langgraph-checkpoint==1.0.12  langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic databricks_langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "how many comments did ITV have on the broadcasted movies?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            "langchain==0.2.16",
            "langchain-community==0.2.16",
            "langgraph-checkpoint==1.0.12",
            "langgraph==0.2.16",
            "pydantic",
            "databricks_langchain", # used for the retriever tool
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "how many comments did ITV have on the broadcasted movies?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "how many comments did ITV have on the broadcasted movies?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out the number of comments on movies broadcasted by ITV. This tool will provide the comment count and percentage for content items across all movies shown by ITV.\n\n<uc_function_call>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"ITV1West.uk20241005092500 +0100\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3444,6.07332428095297],[\\\"ITV1YorkshireWest.uk20241027163000 +0000\\\",\\\"MAMMA MIA! HERE WE GO AGAIN\\\",\\\"ITV\\\",3384,5.96751723773079],[\\\"ITV1West.uk20241229092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3310,5.83702188442344],[\\\"ITV1WestCountry.uk20250104092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3217,5.67302096742907]]}\"}</uc_function_result>\n\nITV had the following comment counts on the broadcasted movies:\n\n1. \"ITV NEWS\" on ITV1 West: 3,444 comments (6.07% of total comments)\n2. \"MAMMA MIA! HERE WE GO AGAIN\" on ITV1 Yorkshire West: 3,384 comments (5.97% of total comments)\n3. \"ITV NEWS\" on ITV1 West: 3,310 comments (5.84% of total comments)\n4. \"ITV NEWS\" on ITV1 West Country: 3,217 comments (5.67% of total comments)"
                },
                {
                    "role": "user",
                    "content": "what about BBC"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "how many comments did ITV have on the broadcasted movies?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out the number of comments on movies broadcasted by ITV. This tool will provide the comment count and percentage for content items across all movies shown by ITV.\n\n<uc_function_call>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"ITV1West.uk20241005092500 +0100\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3444,6.07332428095297],[\\\"ITV1YorkshireWest.uk20241027163000 +0000\\\",\\\"MAMMA MIA! HERE WE GO AGAIN\\\",\\\"ITV\\\",3384,5.96751723773079],[\\\"ITV1West.uk20241229092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3310,5.83702188442344],[\\\"ITV1WestCountry.uk20250104092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3217,5.67302096742907]]}\"}</uc_function_result>\n\nITV had the following comment counts on the broadcasted movies:\n\n1. \"ITV NEWS\" on ITV1 West: 3,444 comments (6.07% of total comments)\n2. \"MAMMA MIA! HERE WE GO AGAIN\" on ITV1 Yorkshire West: 3,384 comments (5.97% of total comments)\n3. \"ITV NEWS\" on ITV1 West: 3,310 comments (5.84% of total comments)\n4. \"ITV NEWS\" on ITV1 West Country: 3,217 comments (5.67% of total comments)"
                },
                {
                    "role": "user",
                    "content": "what about BBC"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.content_comment_count\n\n<uc_function_call>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"BBCThreeHD.uk20241130190000 +0000\\\",\\\"JOHN KRASINSKI: THE STORY SO FAR\\\",\\\"BBC\\\",3393,5.98338829421412],[\\\"CBBC.uk20241128171000 +0000\\\",\\\"TOTAL DRAMARAMA\\\",\\\"BBC\\\",3254,5.73826864408274]]}\"}</uc_function_result>\n\nBBC had the following comment counts on the broadcasted movies:\n\n1. \"JOHN KRASINSKI: THE STORY SO FAR\" on BBC Three HD: 3,393 comments (5.98% of total comments)\n2. \"TOTAL DRAMARAMA\" on CBBC: 3,254 comments (5.74% of total comments)"
                },
                {
                    "role": "user",
                    "content": "how many completed views did bbc get vs ITV?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "how many comments did ITV have on the broadcasted movies?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out the number of comments on movies broadcasted by ITV. This tool will provide the comment count and percentage for content items across all movies shown by ITV.\n\n<uc_function_call>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"ITV1West.uk20241005092500 +0100\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3444,6.07332428095297],[\\\"ITV1YorkshireWest.uk20241027163000 +0000\\\",\\\"MAMMA MIA! HERE WE GO AGAIN\\\",\\\"ITV\\\",3384,5.96751723773079],[\\\"ITV1West.uk20241229092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3310,5.83702188442344],[\\\"ITV1WestCountry.uk20250104092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3217,5.67302096742907]]}\"}</uc_function_result>\n\nITV had the following comment counts on the broadcasted movies:\n\n1. \"ITV NEWS\" on ITV1 West: 3,444 comments (6.07% of total comments)\n2. \"MAMMA MIA! HERE WE GO AGAIN\" on ITV1 Yorkshire West: 3,384 comments (5.97% of total comments)\n3. \"ITV NEWS\" on ITV1 West: 3,310 comments (5.84% of total comments)\n4. \"ITV NEWS\" on ITV1 West Country: 3,217 comments (5.67% of total comments)"
                },
                {
                    "role": "user",
                    "content": "what about BBC"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.content_comment_count\n\n<uc_function_call>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"BBCThreeHD.uk20241130190000 +0000\\\",\\\"JOHN KRASINSKI: THE STORY SO FAR\\\",\\\"BBC\\\",3393,5.98338829421412],[\\\"CBBC.uk20241128171000 +0000\\\",\\\"TOTAL DRAMARAMA\\\",\\\"BBC\\\",3254,5.73826864408274]]}\"}</uc_function_result>\n\nBBC had the following comment counts on the broadcasted movies:\n\n1. \"JOHN KRASINSKI: THE STORY SO FAR\" on BBC Three HD: 3,393 comments (5.98% of total comments)\n2. \"TOTAL DRAMARAMA\" on CBBC: 3,254 comments (5.74% of total comments)"
                },
                {
                    "role": "user",
                    "content": "how many completed views did bbc get vs ITV?"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.avg_watch_per_broadcaster\n\n<uc_function_call>{\"id\":\"zSJzxlJJjc5wLTLCl3uk\",\"name\":\"mzervou.media.avg_watch_per_broadcaster\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"zSJzxlJJjc5wLTLCl3uk\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"broadcaster\\\",\\\"avg_watch_time\\\"],\\\"rows\\\":[[\\\"ITV\\\",50.57572808265329]]}\"}</uc_function_result>\n\nI'm going to use mzervou.media.avg_watch_per_broadcaster\n\n<uc_function_call>{\"id\":\"KXiHNLmimOuIPmDEN0oB\",\"name\":\"mzervou.media.avg_watch_per_broadcaster\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"KXiHNLmimOuIPmDEN0oB\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"broadcaster\\\",\\\"avg_watch_time\\\"],\\\"rows\\\":[[\\\"BBC\\\",50.39427537051703]]}\"}</uc_function_result>\n\nThe average watch time for completed views is as follows:\n\n- ITV: 50.58 minutes\n- BBC: 50.39 minutes\n\nThis indicates that ITV had a slightly higher average watch time compared to BBC."
                },
                {
                    "role": "user",
                    "content": "the title with the most comments for ITV. What do people say in the reviews?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "how many comments did ITV have on the broadcasted movies?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out the number of comments on movies broadcasted by ITV. This tool will provide the comment count and percentage for content items across all movies shown by ITV.\n\n<uc_function_call>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"ITV1West.uk20241005092500 +0100\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3444,6.07332428095297],[\\\"ITV1YorkshireWest.uk20241027163000 +0000\\\",\\\"MAMMA MIA! HERE WE GO AGAIN\\\",\\\"ITV\\\",3384,5.96751723773079],[\\\"ITV1West.uk20241229092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3310,5.83702188442344],[\\\"ITV1WestCountry.uk20250104092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3217,5.67302096742907]]}\"}</uc_function_result>\n\nITV had the following comment counts on the broadcasted movies:\n\n1. \"ITV NEWS\" on ITV1 West: 3,444 comments (6.07% of total comments)\n2. \"MAMMA MIA! HERE WE GO AGAIN\" on ITV1 Yorkshire West: 3,384 comments (5.97% of total comments)\n3. \"ITV NEWS\" on ITV1 West: 3,310 comments (5.84% of total comments)\n4. \"ITV NEWS\" on ITV1 West Country: 3,217 comments (5.67% of total comments)"
                },
                {
                    "role": "user",
                    "content": "what about BBC"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.content_comment_count\n\n<uc_function_call>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"BBCThreeHD.uk20241130190000 +0000\\\",\\\"JOHN KRASINSKI: THE STORY SO FAR\\\",\\\"BBC\\\",3393,5.98338829421412],[\\\"CBBC.uk20241128171000 +0000\\\",\\\"TOTAL DRAMARAMA\\\",\\\"BBC\\\",3254,5.73826864408274]]}\"}</uc_function_result>\n\nBBC had the following comment counts on the broadcasted movies:\n\n1. \"JOHN KRASINSKI: THE STORY SO FAR\" on BBC Three HD: 3,393 comments (5.98% of total comments)\n2. \"TOTAL DRAMARAMA\" on CBBC: 3,254 comments (5.74% of total comments)"
                },
                {
                    "role": "user",
                    "content": "how many completed views did bbc get vs ITV?"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.avg_watch_per_broadcaster\n\n<uc_function_call>{\"id\":\"zSJzxlJJjc5wLTLCl3uk\",\"name\":\"mzervou.media.avg_watch_per_broadcaster\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"zSJzxlJJjc5wLTLCl3uk\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"broadcaster\\\",\\\"avg_watch_time\\\"],\\\"rows\\\":[[\\\"ITV\\\",50.57572808265329]]}\"}</uc_function_result>\n\nI'm going to use mzervou.media.avg_watch_per_broadcaster\n\n<uc_function_call>{\"id\":\"KXiHNLmimOuIPmDEN0oB\",\"name\":\"mzervou.media.avg_watch_per_broadcaster\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"KXiHNLmimOuIPmDEN0oB\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"broadcaster\\\",\\\"avg_watch_time\\\"],\\\"rows\\\":[[\\\"BBC\\\",50.39427537051703]]}\"}</uc_function_result>\n\nThe average watch time for completed views is as follows:\n\n- ITV: 50.58 minutes\n- BBC: 50.39 minutes\n\nThis indicates that ITV had a slightly higher average watch time compared to BBC."
                },
                {
                    "role": "user",
                    "content": "the title with the most comments for ITV. What do people say in the reviews?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out what people are saying in the reviews for the title with the most comments on ITV, which is 'ITV NEWS'.\n\n<uc_function_call>{\"id\":\"mZNHCoCVQUK3ByfNSdZ8\",\"name\":\"mzervou.media.query_reviews\",\"arguments\":\"{\\\"question\\\":\\\"What do people say in the reviews for 'ITV NEWS' on ITV?\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"mZNHCoCVQUK3ByfNSdZ8\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[**Title: A Comprehensive News Roundup - \\\\\\\"ITV News\\\\\\\" Review**\\\\n\\\\n**Rating: 4.5/5**\\\\n\\\\n\\\\\\\"ITV News\\\\\\\" is a thorough and well-structured news program that delivers on its promise of providing national and international news, reports, and analysis from a team of experienced correspondents. The program's format is well-paced, covering a wide range of topics, from politics and current events to sports results.\\\\n\\\\nThe correspondents, based in the UK and around the world, bring a high level of expertise and insight to their reporting, providing in-depth analysis and context to the stories they cover. The program's production quality is also top-notch, with clear and concise editing that makes it easy to follow along.\\\\n\\\\nOne of the strengths of \\\\\\\"ITV News\\\\\\\" is its ability to balance breaking news with more in-depth feature pieces, giving viewers a comprehensive understanding of the issues that matter. The program's correspondents are also not afraid to ask tough questions and hold those in power to account, which adds to the program's credibility and trustworthiness.\\\\n\\\\nThe inclusion of sports results is a nice touch, providing a welcome respite from the often-serious nature of the news. The sports coverage is concise and informative, giving viewers a quick rundown of the latest scores and results.\\\\n\\\\nIf I have any criticisms, it's that the program can sometimes feel a bit dry and formulaic. Some of the segments can feel a bit too long, and the program could benefit from a]\\\"]]}\"}</uc_function_result>\n\nThe reviews for \"ITV NEWS\" on ITV are generally positive, highlighting several key aspects:\n\n- **Comprehensive Coverage**: The program is praised for its thorough and well-structured format, delivering both national and international news with reports and analysis from experienced correspondents.\n\n- **Expert Correspondents**: Reviewers appreciate the high level of expertise and insight provided by the correspondents, who offer in-depth analysis and context to the stories.\n\n- **Production Quality**: The production quality is noted as top-notch, with clear and concise editing that makes the news easy to follow.\n\n- **Balanced Content**: \"ITV News\" is commended for balancing breaking news with in-depth feature pieces, providing a comprehensive understanding of important issues.\n\n- **Credibility and Trustworthiness**: The program is recognized for its credibility, with correspondents not shying away from asking tough questions and holding those in power accountable.\n\n- **Sports Coverage**: The inclusion of sports results is seen as a positive addition, offering a concise and informative rundown of the latest scores.\n\nHowever, some criticisms include the program occasionally feeling dry and formulaic, with some segments being perceived as too long."
                },
                {
                    "role": "user",
                    "content": "what are the demographics of the most active users based on number of interactions"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "how many comments did ITV have on the broadcasted movies?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out the number of comments on movies broadcasted by ITV. This tool will provide the comment count and percentage for content items across all movies shown by ITV.\n\n<uc_function_call>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"VWV6hshSQQB2GpjkEpxC\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"ITV1West.uk20241005092500 +0100\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3444,6.07332428095297],[\\\"ITV1YorkshireWest.uk20241027163000 +0000\\\",\\\"MAMMA MIA! HERE WE GO AGAIN\\\",\\\"ITV\\\",3384,5.96751723773079],[\\\"ITV1West.uk20241229092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3310,5.83702188442344],[\\\"ITV1WestCountry.uk20250104092500 +0000\\\",\\\"ITV NEWS\\\",\\\"ITV\\\",3217,5.67302096742907]]}\"}</uc_function_result>\n\nITV had the following comment counts on the broadcasted movies:\n\n1. \"ITV NEWS\" on ITV1 West: 3,444 comments (6.07% of total comments)\n2. \"MAMMA MIA! HERE WE GO AGAIN\" on ITV1 Yorkshire West: 3,384 comments (5.97% of total comments)\n3. \"ITV NEWS\" on ITV1 West: 3,310 comments (5.84% of total comments)\n4. \"ITV NEWS\" on ITV1 West Country: 3,217 comments (5.67% of total comments)"
                },
                {
                    "role": "user",
                    "content": "what about BBC"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.content_comment_count\n\n<uc_function_call>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"name\":\"mzervou.media.content_comment_count\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"FNLLZ8Mt4pKCD64sVBP2\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"content_id\\\",\\\"title\\\",\\\"broadcaster\\\",\\\"comment_count\\\",\\\"percentage_of_total\\\"],\\\"rows\\\":[[\\\"BBCThreeHD.uk20241130190000 +0000\\\",\\\"JOHN KRASINSKI: THE STORY SO FAR\\\",\\\"BBC\\\",3393,5.98338829421412],[\\\"CBBC.uk20241128171000 +0000\\\",\\\"TOTAL DRAMARAMA\\\",\\\"BBC\\\",3254,5.73826864408274]]}\"}</uc_function_result>\n\nBBC had the following comment counts on the broadcasted movies:\n\n1. \"JOHN KRASINSKI: THE STORY SO FAR\" on BBC Three HD: 3,393 comments (5.98% of total comments)\n2. \"TOTAL DRAMARAMA\" on CBBC: 3,254 comments (5.74% of total comments)"
                },
                {
                    "role": "user",
                    "content": "how many completed views did bbc get vs ITV?"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.avg_watch_per_broadcaster\n\n<uc_function_call>{\"id\":\"zSJzxlJJjc5wLTLCl3uk\",\"name\":\"mzervou.media.avg_watch_per_broadcaster\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"ITV\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"zSJzxlJJjc5wLTLCl3uk\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"broadcaster\\\",\\\"avg_watch_time\\\"],\\\"rows\\\":[[\\\"ITV\\\",50.57572808265329]]}\"}</uc_function_result>\n\nI'm going to use mzervou.media.avg_watch_per_broadcaster\n\n<uc_function_call>{\"id\":\"KXiHNLmimOuIPmDEN0oB\",\"name\":\"mzervou.media.avg_watch_per_broadcaster\",\"arguments\":\"{\\\"input_broadcaster\\\":\\\"BBC\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"KXiHNLmimOuIPmDEN0oB\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"broadcaster\\\",\\\"avg_watch_time\\\"],\\\"rows\\\":[[\\\"BBC\\\",50.39427537051703]]}\"}</uc_function_result>\n\nThe average watch time for completed views is as follows:\n\n- ITV: 50.58 minutes\n- BBC: 50.39 minutes\n\nThis indicates that ITV had a slightly higher average watch time compared to BBC."
                },
                {
                    "role": "user",
                    "content": "the title with the most comments for ITV. What do people say in the reviews?"
                },
                {
                    "role": "assistant",
                    "content": "I need to find out what people are saying in the reviews for the title with the most comments on ITV, which is 'ITV NEWS'.\n\n<uc_function_call>{\"id\":\"mZNHCoCVQUK3ByfNSdZ8\",\"name\":\"mzervou.media.query_reviews\",\"arguments\":\"{\\\"question\\\":\\\"What do people say in the reviews for 'ITV NEWS' on ITV?\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"mZNHCoCVQUK3ByfNSdZ8\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[**Title: A Comprehensive News Roundup - \\\\\\\"ITV News\\\\\\\" Review**\\\\n\\\\n**Rating: 4.5/5**\\\\n\\\\n\\\\\\\"ITV News\\\\\\\" is a thorough and well-structured news program that delivers on its promise of providing national and international news, reports, and analysis from a team of experienced correspondents. The program's format is well-paced, covering a wide range of topics, from politics and current events to sports results.\\\\n\\\\nThe correspondents, based in the UK and around the world, bring a high level of expertise and insight to their reporting, providing in-depth analysis and context to the stories they cover. The program's production quality is also top-notch, with clear and concise editing that makes it easy to follow along.\\\\n\\\\nOne of the strengths of \\\\\\\"ITV News\\\\\\\" is its ability to balance breaking news with more in-depth feature pieces, giving viewers a comprehensive understanding of the issues that matter. The program's correspondents are also not afraid to ask tough questions and hold those in power to account, which adds to the program's credibility and trustworthiness.\\\\n\\\\nThe inclusion of sports results is a nice touch, providing a welcome respite from the often-serious nature of the news. The sports coverage is concise and informative, giving viewers a quick rundown of the latest scores and results.\\\\n\\\\nIf I have any criticisms, it's that the program can sometimes feel a bit dry and formulaic. Some of the segments can feel a bit too long, and the program could benefit from a]\\\"]]}\"}</uc_function_result>\n\nThe reviews for \"ITV NEWS\" on ITV are generally positive, highlighting several key aspects:\n\n- **Comprehensive Coverage**: The program is praised for its thorough and well-structured format, delivering both national and international news with reports and analysis from experienced correspondents.\n\n- **Expert Correspondents**: Reviewers appreciate the high level of expertise and insight provided by the correspondents, who offer in-depth analysis and context to the stories.\n\n- **Production Quality**: The production quality is noted as top-notch, with clear and concise editing that makes the news easy to follow.\n\n- **Balanced Content**: \"ITV News\" is commended for balancing breaking news with in-depth feature pieces, providing a comprehensive understanding of important issues.\n\n- **Credibility and Trustworthiness**: The program is recognized for its credibility, with correspondents not shying away from asking tough questions and holding those in power accountable.\n\n- **Sports Coverage**: The inclusion of sports results is seen as a positive addition, offering a concise and informative rundown of the latest scores.\n\nHowever, some criticisms include the program occasionally feeling dry and formulaic, with some segments being perceived as too long."
                },
                {
                    "role": "user",
                    "content": "what are the demographics of the most active users based on number of interactions"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use mzervou.media.ask_genie_media\n\n<uc_function_call>{\"id\":\"9yJwnqF4IDPRbZFH9krj\",\"name\":\"mzervou.media.ask_genie_media\",\"arguments\":\"{\\\"question\\\":\\\"What are the demographics of the most active users based on the number of interactions?\\\",\\\"contextual_history\\\":\\\"no relevant history\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9yJwnqF4IDPRbZFH9krj\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Genie Results are: \\\\nSpace ID: 01efd80d29511dfdb8cdbcad56e36ddf\\\\nConversation ID: 01efd97cbc721c2085f6d6ef60e5adec\\\\nQuestion That Was Asked: Use the contextual history to answer the question. The history may or may not help you. Use it if you find it relevant.\\\\n        \\\\n        Contextual History: no relevant history\\\\n        \\\\n        Question to answer: What are the demographics of the most active users based on the number of interactions?\\\\n        \\\\nContent: None\\\\nSQL Query: WITH user_interaction_counts AS ( \\\\n SELECT `user_id`, COUNT(*) AS interaction_count \\\\n FROM `mzervou`.`media`.`user_interactions` \\\\n WHERE `user_id` IS NOT NULL \\\\n GROUP BY `user_id` \\\\n ORDER BY interaction_count DESC \\\\n LIMIT 10 \\\\n) \\\\nSELECT uic.`user_id`, up.`gender`, up.`age`, up.`subscription` \\\\nFROM user_interaction_counts uic \\\\nJOIN `mzervou`.`media`.`user_profiles` up \\\\nON uic.`user_id` = up.`user_id` \\\\nWHERE up.`user_id` IS NOT NULL;\\\\nSQL Query Description: This analysis identifies the top 10 users based on their interaction counts with media content. It then provides demographic details, including gender, age, and subscription status, for these users by combining interaction data with user profile information.\\\\nSQL Query Result: [{'user_id': 'user_49611', 'gender': 'Male', 'age': 64, 'subscription': 'Free'}, {'user_id': 'user_99964', 'gender': 'Female', 'age': 51, 'subscription': 'Premium'}, {'user_id': 'user_55476', 'gender': 'Male', 'age': 50, 'subscription': 'Free'}, {'user_id': 'user_56078', 'gender': 'Female', 'age': 21, 'subscription': 'Free'}, {'user_id': 'user_14634', 'gender': 'Female', 'age': 36, 'subscription': 'Premium'}, {'user_id': 'user_25578', 'gender': 'Male', 'age': 22, 'subscription': 'Premium'}, {'user_id': 'user_26484', 'gender': 'Female', 'age': 48, 'subscription': 'Free'}, {'user_id': 'user_61894', 'gender': 'Male', 'age': 27, 'subscription': 'Free'}, {'user_id': 'user_41049', 'gender': 'Female', 'age': 56, 'subscription': 'Premium'}, {'user_id': 'user_71708', 'gender': 'Female', 'age': 59, 'subscription': 'Free'}]\\\\nError: None\\\"]]}\"}</uc_function_result>\n\nThe demographics of the most active users based on the number of interactions are as follows:\n\n1. **User ID: user_49611**\n   - Gender: Male\n   - Age: 64\n   - Subscription: Free\n\n2. **User ID: user_99964**\n   - Gender: Female\n   - Age: 51\n   - Subscription: Premium\n\n3. **User ID: user_55476**\n   - Gender: Male\n   - Age: 50\n   - Subscription: Free\n\n4. **User ID: user_56078**\n   - Gender: Female\n   - Age: 21\n   - Subscription: Free\n\n5. **User ID: user_14634**\n   - Gender: Female\n   - Age: 36\n   - Subscription: Premium\n\n6. **User ID: user_25578**\n   - Gender: Male\n   - Age: 22\n   - Subscription: Premium\n\n7. **User ID: user_26484**\n   - Gender: Female\n   - Age: 48\n   - Subscription: Free\n\n8. **User ID: user_61894**\n   - Gender: Male\n   - Age: 27\n   - Subscription: Free\n\n9. **User ID: user_41049**\n   - Gender: Female\n   - Age: 56\n   - Subscription: Premium\n\n10. **User ID: user_71708**\n    - Gender: Female\n    - Age: 59\n    - Subscription: Free\n\nThese users represent a mix of genders, ages, and subscription types, with both free and premium users being highly active."
                },
                {
                    "role": "user",
                    "content": "given that we are investing in more comedy time slots, is this a good choice? what does marketing research report says?"
                }
            ]
        },
        "expected_response": None
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "main"
schema = "default"
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

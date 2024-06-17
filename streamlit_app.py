!pip install google-cloud-aiplatform --upgrade --user -q
!pip install --upgrade langchain -q
!pip install streamlit -q
!pip install langchain-google-genai -q
!pip install langchain_community -q

def load_df(table_name):
   from google.cloud import bigquery
   project_id='cloud-s2-172915'
   client=bigquery.Client(project=project_id)

   dataset_ref=client.dataset("brand_pulse_tracker",project=project_id)
   dataset=client.get_dataset(dataset_ref)

   table_ref=dataset_ref.table(table_name)
   table = client.get_table(table_ref)
   df=client.list_rows(table).to_dataframe()
   return df


df=load_df("parent_brand_text_to_sql")

import numpy as np
import pandas as pd
import sqlite3

conn=sqlite3.connect("brand_pulse.sqlite")
c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS text_to_sql (CompanySize text, Brand_Name text, \
                                                                          Region text, Country text, Industry text,\
                                                                           Buyer_Segment_Rollup text, Role text, Pulse int,\
                                                                           KPI_Type text, KPI_Subtype text, Sample_Size float,\
                                                                         KPI_Numerator float, KPI_Denominator float)")
#c.execute("CREATE TABLE IF NOT EXISTS text_to_sql (CompanySize text, CompanyType text, Region text,Brand_Name text, \
#                                                   Country text, Industry text, Buyer_Segment_Rollup text, Role text, Pulse int, KPI_Type text,\
#                                                   KPI_Subtype text, Current_Customer_Prospect text, Sample_Size float, KPI_Numerator float,\
#                                                   KPI_Denominator float)")
conn.commit()
df.to_sql("text_to_sql", conn, if_exists='replace', index = False)


import streamlit as st
import ast
from bigframes import dataframe
import vertexai

from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents import AgentType, create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
#from langchain_google_vertexai import ChatVertexAI
from prompt_text_to_sql_bp import Prompt_Text
import getpass
import os
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI

final_prompt = Prompt_Text("question")

#PROJECT_ID = "sysomosapi2"  # @param {type:"string"}
#vertexai.init(project=PROJECT_ID, location="us-central1")

input_db = SQLDatabase.from_uri('sqlite:///brand_pulse.sqlite')

os.environ["GOOGLE_API_KEY"]= "AIzaSyCB0IkCHaRJKZqeH9y2ym1p8ThW1eEmzhY"

#gemini_pro_llm = ChatVertexAI(model_name="gemini-pro",temperature=0,max_output_tokens=1000)
gemini_pro_llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0)


sql_toolkit = SQLDatabaseToolkit(db=input_db, llm=gemini_pro_llm)
sql_toolkit.get_tools()

sqldb_agent = create_sql_agent(
    llm=gemini_pro_llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True}

)


st.set_page_config(page_title="Text-to-SQL AI Assistant",
                       page_icon=":sparkles:",layout="wide")

st.header("BRAND PULSE DATA ASSISTANT :robot_face:")
st.subheader("I am a Text-to-SQL chatbot. I am capable of querying the brand pulse database \
and answer your questions", divider= "rainbow")
st.divider()
st.write(":star:Note: For best results, ask me specific questions one at a time. Currently I \
am capable of answering parent brand level metrics. Workload level metrics is currently under development :hammer_and_wrench:")
st.divider()
st.subheader("Enter your question here:")
user_question = st.text_input("")
if st.button("Submit"):
  if user_question:
    with st.spinner("Processing"):
      st.write("Peek into my brain below:")
      st_callback = StreamlitCallbackHandler(st.container()) #Displays what is going on in the background
      response = sqldb_agent.run(final_prompt.format(question=user_question),callbacks=[st_callback])
      st.write(response)



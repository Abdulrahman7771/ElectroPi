# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:21:53 2023

@author: 3ndalib
"""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import mpld3
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

db_connection_str = 'mysql+pymysql://root:@localhost:3306/mysql'
db_connection = create_engine(db_connection_str)
query ="SELECT registration_date FROM users"
RDates = pd.read_sql(query, con=db_connection)
query ="SELECT subscription_date FROM users where subscription_date IS NOT NULL"
SDates = pd.read_sql(query, con=db_connection)
query ="SELECT bundle_name FROM bundles where user_id IS NOT NULL"
BUDates = pd.read_sql(query, con=db_connection)
SDates['subscription_date'] = pd.to_datetime(SDates['subscription_date'])
fig = plt.figure(figsize=(7,5)) 
print(RDates.info())
sns.histplot(RDates["registration_date"],label= "Registration")
sns.histplot(SDates["subscription_date"],label="Subscription")

#sns.countplot(data=RDates,x="registration_date")
#plt.bar(RDates)
plt.show()
plt.legend()

print(SDates.max())
#st.pyplot(fig)
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html,height=600,width=1000)

query ="SELECT user_id FROM users where 10k_AI_initiative = 1"
TKUsers = pd.read_sql(query, con=db_connection)

TKUsers["CompletedCoursesCount"] = None
#SELECT COUNT(DISTINCT column_name) FROM table_name

for i in range(TKUsers.shape[0]):
    lquery = f"""SELECT DISTINCT course_id as LastCourse,course_degree as LastDegree, completion_date as LastDate
    FROM user_completed_courses WHERE user_id = {TKUsers.iloc[i,0]} """
    
    query = f"SELECT COUNT(DISTINCT course_id) FROM user_completed_courses where user_id = {TKUsers.iloc[i,0]}"
    TKUsers.iloc[i,1] = pd.read_sql(query, con=db_connection)
    #TKUsers._append(pd.read_sql(lquery, con=db_connection))
    TKUsers._append((pd.read_sql(lquery, con=db_connection)).sort_values(['LastCourse', 'LastDate']).drop_duplicates('LastDate', keep='last'))


st.dataframe(TKUsers)

























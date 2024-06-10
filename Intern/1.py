# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:21:53 2023

@author: 3ndalib
"""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import pandas as pd

db_connection_str = 'mysql+pymysql://root:@localhost:3306/mysql'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM admins', con=db_connection)
print(df)


































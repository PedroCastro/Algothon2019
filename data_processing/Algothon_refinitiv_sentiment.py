# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:08:38 2019

@author: alexm
"""

# Sentiment
import google
import pandas as pd
import numpy as np
from google.oauth2 import service_account
import pandas_gbq
import json
import seaborn as sns
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
from elasticsearch import Elasticsearch

warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('seaborn')
pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', -1)
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))



personal_cred = json.loads('{"type":"service_account","project_id":"tr-data-workbench","private_key_id":"c76db9c8c609e33ca859845b187289a347cb2ab9","private_key":"-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCjIYnqAHAc04to\\n5GeHBuYd1wzJ/ITnoAGkLfdPurOS1gWAz18f7rjXtKCPx4RT0dCBWwQYqcABSVHw\\nT1a2ei1cAaKFX/uDcOuZky7VZ0Imv+3YOJgvpvO7jZAwaeq3fNHCMPILsQQIzZyJ\\n+WCdsRNi51eZUfKyhw8bAE+U86k0WZI1jAX0RlUxEA0JOMmlZVLkOezlEGp11Y97\\nD/9eZv3RsT7AfAnYkSES4NhFrMyv9ilLkuaa4wMQePzgRXhHte8UkXtXnnxrSNJy\\nJV5K2G9jSTmTwg5iAVbQSTOOgNL8pQOKVhx76LukrjGqDkzXfWQ6Y2+R7DIVFdmU\\ngEasY/KHAgMBAAECggEAB54F5snu2KaKiIy8+Z6GP94OODkbXgZBrV0cHDpdyKOu\\nJ03UZXuFlBG+7mSYA2+O3eUJgIe32H3ukcF5DUiqAsIGYokqoTOa1lWVRtyG4T2H\\n3KQ5e8tYiQLbBcSCwKvXNTGJGKfcxioO5WTVcLRkPfNxfefh5LMCLvX8jiCZxdqJ\\nxMgkXFruDMVbhEBS0WxY3C0ic4vAz4ibSvG3ZHw+/NjCoJcfNqWEOOrXsdKdVtki\\nLqCIQp2+4JihQWdY5NRcqJewGvV0fEV0eUVFskuHD2XRlCZS1pmfOcUgVi1NHeZb\\nbxMiQLuxUMi0ODi8JlyHRkgYR7upCVOLY2G5v5oWKQKBgQDVWrRPVQs7+FnHUlA+\\nuKro4S4Hgld/+bFfGD7vJhMTEUEmXCg1a8ZOClhsLntF65thyo9WDrBV6QayV2cM\\n3teu/1paX1IBe+rX0USLoDqaS2yWhLPRfhVB8AdfIaSTt8cGP8HW4EBvWL9MTuZ5\\nY///gci0s3wHaLUNqQb9OtauOwKBgQDDvOsgRQslscJ6hD5mZ9IQjtnASVMLgSjB\\nHNjdgih+fqxFL0tuN+UaVnXMuRbtbDoIypZFWK7ULOzbjjj1jxOywq+mdk1/Qf1L\\nqZ3EocfUrmIrIlZ7p4kD+nqBWPbO2pHAieRJCEMwXcRILNkgRYG0/ssRnc/AyM45\\nWKZ7pHYMJQKBgQDUlgV1ysJf5ezm+3CznmPFmaG4n7o57P08SLdkqSZ2aEnnRApY\\neGPmnM5QNNxl5gY0IZZC5G31nDQs/YPTwjNczlkkFThr/CIbGwxWp7fcx+yR6fYW\\ndrANvHJL6wTGn2azJlIpndb2W5J5IWDqcabB23q1+uVJqJ5G1zX8mmUQwwKBgQCP\\n2zrbXqMQsxKBxMuvq8IBlVuILNux6t0vAKqKMezc+vBVcKr5eG8S6lRtf+LP3+jP\\nKUVD9ieXnOT/gAlwwBT0Ho3Fw9C1JKSqhSCEsXoSX4+asAProXfbyq1afy31XUId\\nxbpXypDG7UMi4IM7aponkdNhQSC9SVf3YaYJ3Rc9WQKBgAIcmp0STmO/KHC91IoL\\nq/bYLHiuzTioyOWj4UDq8OsYjdR1xqVqPqtTAHi4UeBkH+P1jNZiCne+DHlm8BXL\\n7PRoC5k+YeMeC66zr9WgzvD6rUaSgkss22ccQMDk246oWI9/dLXbovVNnWwhGBsm\\n9u0MLuqioMHONOEOMAFb5git\\n-----END PRIVATE KEY-----\\n","client_email":"spd7hiag02t8nvfc03ta0qk3k2s7oh@tr-data-workbench.iam.gserviceaccount.com","client_id":"107033979330412125634","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/spd7hiag02t8nvfc03ta0qk3k2s7oh%40tr-data-workbench.iam.gserviceaccount.com"}') # your personal key for Tick History on BigQuery
credentials = service_account.Credentials.from_service_account_info(personal_cred)

ELASTICSEARCH_API_KEY = 'sQw3OvDNhs3beM9fCwzLCQtMRBPbgifC1Y6Rotj0' # your personal key for Data Science Accelerator access to News on Elasticsearch
ELASTICSEARCH_HOST = 'dsa-stg-newsarchive-api.fr-nonprod.aws.thomsonreuters.com'

# Creating Elasticsearch connection through api gateway
es = Elasticsearch(
    host=ELASTICSEARCH_HOST,
    port=443,
    headers={'X-api-key': ELASTICSEARCH_API_KEY}, 
    use_ssl=True, 
    timeout=30
)

index_name = 'newsarchive'

sql_qry = """
    SELECT 
        * 
    FROM 
        TRNA.news 
    WHERE 
        id = 'tr:L2N1RH1C6_1804046gbF7v5pWAYkV91d03Vvjmsye7cwgu5B6H5qlv-4297297477'
"""
trna_news_df_single_row = pandas_gbq.read_gbq(sql_qry, project_id="tr-data-workbench", credentials=credentials, dialect='standard')


import json
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# Data Science Accelerator Credentials 
personal_cred = json.loads('{"type":"service_account","project_id":"tr-data-workbench","private_key_id":"c76db9c8c609e33ca859845b187289a347cb2ab9","private_key":"-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCjIYnqAHAc04to\\n5GeHBuYd1wzJ/ITnoAGkLfdPurOS1gWAz18f7rjXtKCPx4RT0dCBWwQYqcABSVHw\\nT1a2ei1cAaKFX/uDcOuZky7VZ0Imv+3YOJgvpvO7jZAwaeq3fNHCMPILsQQIzZyJ\\n+WCdsRNi51eZUfKyhw8bAE+U86k0WZI1jAX0RlUxEA0JOMmlZVLkOezlEGp11Y97\\nD/9eZv3RsT7AfAnYkSES4NhFrMyv9ilLkuaa4wMQePzgRXhHte8UkXtXnnxrSNJy\\nJV5K2G9jSTmTwg5iAVbQSTOOgNL8pQOKVhx76LukrjGqDkzXfWQ6Y2+R7DIVFdmU\\ngEasY/KHAgMBAAECggEAB54F5snu2KaKiIy8+Z6GP94OODkbXgZBrV0cHDpdyKOu\\nJ03UZXuFlBG+7mSYA2+O3eUJgIe32H3ukcF5DUiqAsIGYokqoTOa1lWVRtyG4T2H\\n3KQ5e8tYiQLbBcSCwKvXNTGJGKfcxioO5WTVcLRkPfNxfefh5LMCLvX8jiCZxdqJ\\nxMgkXFruDMVbhEBS0WxY3C0ic4vAz4ibSvG3ZHw+/NjCoJcfNqWEOOrXsdKdVtki\\nLqCIQp2+4JihQWdY5NRcqJewGvV0fEV0eUVFskuHD2XRlCZS1pmfOcUgVi1NHeZb\\nbxMiQLuxUMi0ODi8JlyHRkgYR7upCVOLY2G5v5oWKQKBgQDVWrRPVQs7+FnHUlA+\\nuKro4S4Hgld/+bFfGD7vJhMTEUEmXCg1a8ZOClhsLntF65thyo9WDrBV6QayV2cM\\n3teu/1paX1IBe+rX0USLoDqaS2yWhLPRfhVB8AdfIaSTt8cGP8HW4EBvWL9MTuZ5\\nY///gci0s3wHaLUNqQb9OtauOwKBgQDDvOsgRQslscJ6hD5mZ9IQjtnASVMLgSjB\\nHNjdgih+fqxFL0tuN+UaVnXMuRbtbDoIypZFWK7ULOzbjjj1jxOywq+mdk1/Qf1L\\nqZ3EocfUrmIrIlZ7p4kD+nqBWPbO2pHAieRJCEMwXcRILNkgRYG0/ssRnc/AyM45\\nWKZ7pHYMJQKBgQDUlgV1ysJf5ezm+3CznmPFmaG4n7o57P08SLdkqSZ2aEnnRApY\\neGPmnM5QNNxl5gY0IZZC5G31nDQs/YPTwjNczlkkFThr/CIbGwxWp7fcx+yR6fYW\\ndrANvHJL6wTGn2azJlIpndb2W5J5IWDqcabB23q1+uVJqJ5G1zX8mmUQwwKBgQCP\\n2zrbXqMQsxKBxMuvq8IBlVuILNux6t0vAKqKMezc+vBVcKr5eG8S6lRtf+LP3+jP\\nKUVD9ieXnOT/gAlwwBT0Ho3Fw9C1JKSqhSCEsXoSX4+asAProXfbyq1afy31XUId\\nxbpXypDG7UMi4IM7aponkdNhQSC9SVf3YaYJ3Rc9WQKBgAIcmp0STmO/KHC91IoL\\nq/bYLHiuzTioyOWj4UDq8OsYjdR1xqVqPqtTAHi4UeBkH+P1jNZiCne+DHlm8BXL\\n7PRoC5k+YeMeC66zr9WgzvD6rUaSgkss22ccQMDk246oWI9/dLXbovVNnWwhGBsm\\n9u0MLuqioMHONOEOMAFb5git\\n-----END PRIVATE KEY-----\\n","client_email":"spd7hiag02t8nvfc03ta0qk3k2s7oh@tr-data-workbench.iam.gserviceaccount.com","client_id":"107033979330412125634","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/spd7hiag02t8nvfc03ta0qk3k2s7oh%40tr-data-workbench.iam.gserviceaccount.com"}')# your personal key for BigQuery
proj = u'tr-data-workbench'
cred = service_account.Credentials.from_service_account_info(personal_cred)

ticker = 'AAPL'

query_daily = """
    SELECT 
        windowTimestamp,
        ticker,
        sentiment,
        priceDirection
    FROM 
        `tr-data-workbench.TRMI.daily` 
"""

query_params = [
    bigquery.ScalarQueryParameter('ticker', 'STRING', ticker)
]


job_config = bigquery.QueryJobConfig()
job_config.query_parameters = query_params
job_config.use_legacy_sql = False
bigquery_client = bigquery.Client(project=proj, credentials=cred)
df_daily = bigquery_client.query(query_daily, job_config=job_config).to_dataframe()

df_daily.head()

df_daily.to_csv('Refinitiv_sentiment.csv')





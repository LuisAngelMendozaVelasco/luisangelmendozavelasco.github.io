# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: tensorflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chicago Indicators

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Data_Engineering_Foundations_Specialization/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Analyze real-world datasets from the city of Chicago using SQLite statements.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import sqlite3
import folium
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
con = sqlite3.connect(":memory:")
cur = con.cursor()

# %%
census_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCensusData.csv"
schools_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoPublicSchools.csv"
crime_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCrimeData.csv"

df_census = pd.read_csv(census_url)
df_schools = pd.read_csv(schools_url)
df_crime = pd.read_csv(crime_url)

df_census.to_sql("CENSUS_DATA", con, if_exists='replace', index=False, method="multi")
df_schools.to_sql("CHICAGO_PUBLIC_SCHOOLS", con, if_exists='replace', index=False, method="multi")
df_crime.to_sql("CHICAGO_CRIME_DATA", con, if_exists='replace', index=False, method="multi");

# %%
df_census.head()

# %%
df_schools.head()

# %%
df_crime.head()

# %% [markdown]
# ## Understand the dataset
#
# There are three datasets available on the [Chicago Data Portal](https://data.cityofchicago.org/):
#
# ### 1. Socioeconomic Indicators in Chicago
#
# This dataset contains a selection of six socioeconomic indicators of public health importance and a "hardship index" for each community area in Chicago, covering the years 2008 to 2012.
#
# A detailed description of this dataset and the original dataset can be obtained from the Chicago Data Portal at this [link](https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-C/kn9c-c2s2).

# %%
df_census.info()

# %% [markdown]
# ### 2. Chicago Public Schools
#
# This dataset shows all school performance data used to generate CPS School Report Cards for the 2011-2012 school year.
#
# A detailed description of this dataset and the original dataset can be obtained from the Chicago Data Portal at this [link](https://data.cityofchicago.org/Education/Chicago-Public-Schools-Progress-Report-Cards-2011-/9xs2-f89t).

# %%
df_schools.info()

# %% [markdown]
# ### 3. Chicago Crime Data
#
# This dataset reflects reported crime incidents (excluding murders) that occurred in the City of Chicago from 2001 to 2012.
#
# A detailed description of this dataset and the original dataset can be obtained from the Chicago Data Portal at this [link](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2).

# %%
df_crime.info()

# %% [markdown]
# ## Visualize the location of public schools and where crimes occurred

# %% [markdown]
# The red markers indicate where crimes occurred, while the blue markers indicate the location of public schools.

# %%
map = folium.Map(location=[41.88, -87.62], zoom_start=12)
feature_group = folium.FeatureGroup()

for point in list(df_schools[["Latitude", "Longitude"]].dropna().to_numpy()):
    feature_group.add_child(folium.Marker(point, popup='School', icon=folium.Icon(color='blue')))

for point in list(df_crime[["LATITUDE", "LONGITUDE"]].dropna().to_numpy()):
    feature_group.add_child(folium.Marker(point, popup='Crime', icon=folium.Icon(color='red')))

map.add_child(feature_group)
map

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### What percentage of community areas in Chicago have a hardship index greater than 50?

# %%
query = """
            SELECT (COUNT(CASE WHEN HARDSHIP_INDEX > 50 THEN 1 END) * 100 / COUNT(*)) || '%' AS Percentage
            FROM CENSUS_DATA
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the community areas with the highest and lowest hardship index?

# %%
query = """
            SELECT COMMUNITY_AREA_NAME,  HARDSHIP_INDEX
            FROM CENSUS_DATA
            WHERE HARDSHIP_INDEX IN ((SELECT MAX(HARDSHIP_INDEX) FROM CENSUS_DATA),
                                     (SELECT MIN(HARDSHIP_INDEX) FROM CENSUS_DATA))
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### Which Chicago community areas have per-capita incomes greater than $50,000?

# %%
query = """
            SELECT COMMUNITY_AREA_NAME, PER_CAPITA_INCOME
            FROM CENSUS_DATA
            WHERE PER_CAPITA_INCOME > 50000
            ORDER BY PER_CAPITA_INCOME DESC
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What is the relationship between the `PER_CAPITA_INCOME` and `HARDSHIP_INDEX` features?

# %%
query = """
            SELECT PER_CAPITA_INCOME, HARDSHIP_INDEX
            FROM CENSUS_DATA
        """

plt.figure()
sns.scatterplot(x="PER_CAPITA_INCOME", y="HARDSHIP_INDEX", data=pd.read_sql_query(query, con))
plt.show()

# %% [markdown]
# ### How many schools of each type are in the dataset?

# %%
query = """
            SELECT "Elementary, Middle, or High School", COUNT(*) 
            FROM CHICAGO_PUBLIC_SCHOOLS 
            GROUP BY "Elementary, Middle, or High School"
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### Which schools have the highest and lowest safety scores?

# %%
query = """
            SELECT NAME_OF_SCHOOL, SAFETY_SCORE
            FROM CHICAGO_PUBLIC_SCHOOLS
            WHERE SAFETY_SCORE IN ((SELECT MAX(SAFETY_SCORE) FROM CHICAGO_PUBLIC_SCHOOLS),
                                   (SELECT MIN(SAFETY_SCORE) FROM CHICAGO_PUBLIC_SCHOOLS))
            ORDER BY SAFETY_SCORE ASC
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the five schools with the lowest safety scores?

# %%
query = """
            SELECT NAME_OF_SCHOOL, SAFETY_SCORE 
            FROM CHICAGO_PUBLIC_SCHOOLS
            WHERE SAFETY_SCORE IS NOT NULL
            ORDER BY SAFETY_SCORE ASC
            LIMIT 5
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the five schools with the highest average student attendance?

# %%
query = """ 
            SELECT NAME_OF_SCHOOL, AVERAGE_STUDENT_ATTENDANCE
            FROM CHICAGO_PUBLIC_SCHOOLS
            ORDER BY AVERAGE_STUDENT_ATTENDANCE DESC
            LIMIT 5
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### Which schools have average student attendance lower than 70%?

# %%
query = """ 
            SELECT NAME_OF_SCHOOL, AVERAGE_STUDENT_ATTENDANCE
            FROM CHICAGO_PUBLIC_SCHOOLS
            WHERE AVERAGE_STUDENT_ATTENDANCE < 70
            ORDER BY AVERAGE_STUDENT_ATTENDANCE
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What is the hardship index for the five community areas with the lowest college enrollment?

# %%
query = """ 
            SELECT CENSUS_DATA.COMMUNITY_AREA_NAME, HARDSHIP_INDEX, SUM(COLLEGE_ENROLLMENT)
            FROM CENSUS_DATA, CHICAGO_PUBLIC_SCHOOLS
            WHERE CENSUS_DATA.COMMUNITY_AREA_NUMBER = CHICAGO_PUBLIC_SCHOOLS.COMMUNITY_AREA_NUMBER
            GROUP BY CENSUS_DATA.COMMUNITY_AREA_NAME
            ORDER BY SUM(COLLEGE_ENROLLMENT) ASC
            LIMIT 5
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What is the hardship index for the five community areas with the highest total college enrollment?

# %%
query = """ 
            SELECT CENSUS_DATA.COMMUNITY_AREA_NAME, HARDSHIP_INDEX, SUM(COLLEGE_ENROLLMENT)
            FROM CENSUS_DATA, CHICAGO_PUBLIC_SCHOOLS
            WHERE CENSUS_DATA.COMMUNITY_AREA_NUMBER = CHICAGO_PUBLIC_SCHOOLS.COMMUNITY_AREA_NUMBER
            GROUP BY CENSUS_DATA.COMMUNITY_AREA_NAME
            ORDER BY SUM(COLLEGE_ENROLLMENT) DESC
            LIMIT 5
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the community areas with per-capita income less than $11,000?

# %%
query = """
            SELECT COMMUNITY_AREA_NAME, PER_CAPITA_INCOME
            FROM CENSUS_DATA
            WHERE PER_CAPITA_INCOME < 11000
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the case numbers for crimes involving minors?

# %%
query = """
            SELECT CASE_NUMBER, DESCRIPTION
            FROM CHICAGO_CRIME_DATA
            WHERE DESCRIPTION LIKE '%MINOR%'     
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the kidnapping crimes involving a child?

# %%
query = """
            SELECT CASE_NUMBER, PRIMARY_TYPE, DESCRIPTION
            FROM CHICAGO_CRIME_DATA
            WHERE PRIMARY_TYPE == "KIDNAPPING" AND DESCRIPTION LIKE '%CHILD%'    
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What kind of crimes were recorded in schools?

# %%
query = """
            SELECT PRIMARY_TYPE, LOCATION_DESCRIPTION
            FROM CHICAGO_CRIME_DATA
            WHERE LOCATION_DESCRIPTION LIKE '%SCHOOL%'   
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What is the average safety score for each type of school?

# %%
query = """
            SELECT "Elementary, Middle, or High School", AVG(SAFETY_SCORE)
            FROM CHICAGO_PUBLIC_SCHOOLS
            GROUP BY "Elementary, Middle, or High School"
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### What are the five community areas with the highest percentage of households below the poverty line?

# %%
query = """
            SELECT COMMUNITY_AREA_NAME, PERCENT_HOUSEHOLDS_BELOW_POVERTY
            FROM CENSUS_DATA
            ORDER BY PERCENT_HOUSEHOLDS_BELOW_POVERTY DESC 
            LIMIT 5
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### Which community areas are the most crime prone?

# %%
query = """
            SELECT CENSUS_DATA.COMMUNITY_AREA_NAME, COUNT(CHICAGO_CRIME_DATA.COMMUNITY_AREA_NUMBER) AS FREQUENCY
            FROM CHICAGO_CRIME_DATA, CENSUS_DATA
            WHERE CHICAGO_CRIME_DATA.COMMUNITY_AREA_NUMBER == CENSUS_DATA.COMMUNITY_AREA_NUMBER
            GROUP BY CHICAGO_CRIME_DATA.COMMUNITY_AREA_NUMBER
            ORDER BY FREQUENCY DESC 
            LIMIT 5
        """

pd.read_sql_query(query, con)

# %% [markdown]
# ### Which community areas have the lowest number of crimes?

# %%
query = """
            SELECT CENSUS_DATA.COMMUNITY_AREA_NAME, COUNT(CHICAGO_CRIME_DATA.COMMUNITY_AREA_NUMBER) AS FREQUENCY
            FROM CHICAGO_CRIME_DATA, CENSUS_DATA
            WHERE CHICAGO_CRIME_DATA.COMMUNITY_AREA_NUMBER == CENSUS_DATA.COMMUNITY_AREA_NUMBER
            GROUP BY CHICAGO_CRIME_DATA.COMMUNITY_AREA_NUMBER
            ORDER BY FREQUENCY ASC
            LIMIT 5
        """

pd.read_sql_query(query, con)

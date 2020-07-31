#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# plotly is interactive visualization - 2 modes online and offline
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[2]:


covid19_confirmed = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/time_series_covid19_confirmed_global.csv')
covid19_deaths = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/time_series_covid19_deaths_global.csv')
covid19_recovered = pd.read_csv('https://raw.githubusercontent.com/ammishra08/COVID-19/master/covid_19_datasets/covid_19_globaldata/time_series_covid19_recovered_global.csv')


# In[3]:


covid19_confirmed.head(20)


# In[4]:


covid19_confirmed.rename({'Province/State':'State','Country/Region':'Country'}, axis =1,inplace=True)
covid19_deaths.rename({'Province/State':'State','Country/Region':'Country'}, axis =1,inplace=True)
covid19_recovered.rename({'Province/State':'State','Country/Region':'Country'}, axis =1,inplace=True)


# In[5]:


covid19_confirmed.head()


# In[6]:


covid_country = covid19_confirmed.drop(['State','Lat','Long'], axis =1)


# In[7]:


covid_country


# In[8]:


covid_country = covid_country.groupby(['Country']).sum()


# In[9]:


covid_country.head(10)


# In[10]:


covid_country.iloc[:,-1].sum()


# ## Visualization

# In[11]:


sns.set_style('darkgrid')
covid_country.sort_values(by = covid_country.columns[-1], ascending = False).head(10).transpose().plot(figsize=(15,8), fontsize = 15)
plt.title('Covid-19 Time Series Visuaization of Confirmed Cases',fontsize=25)
plt.show()


# ## Plotly Visualization

# In[12]:


covid_country.sort_values(by = covid_country.columns[-1], ascending = False).head(10).transpose().iplot()


# In[13]:


covid_country.loc['India'].transpose().iplot(title = "Time Series Confirmed Cases in India", color='orange', yTitle = 'No. of Cases')


# In[14]:


covid_country.loc['India'].diff().iplot(title = "Daily Change in Confirmed Cases in India")


# ## Plotting Covid-19 cases on World Map

# In[15]:


import folium


# In[16]:


covid19_confirmed


# In[17]:


world_map = folium.Map(location = [10,0], zoom_start=2, max_zoom=8, min_zoom=1, width='100%', tiles = 'CartoDB dark_matter')
for i in range(0, len(covid19_confirmed)):
    folium.Circle(location=[covid19_confirmed.iloc[i]['Lat'], covid19_confirmed.iloc[i]['Long']],
                 radius = (int((np.log(covid19_confirmed.iloc[i,-1]+1.00001)))+0.2)*40000, color = 'red', fill = True,
                 tooltip = "<h5 style='text-align:center;font-weight:bold'>" + covid19_confirmed.iloc[i]['Country']
                 + "</h5>" + "<li>Confirmed " +str(covid19_confirmed.iloc[i,-1]) + "</li>").add_to(world_map)

world_map


# In[18]:


import gmaps


# In[19]:


covid19_confirmed.iloc[0:len(covid19_confirmed),-1]


# In[20]:


get_ipython().system('jupyter nbextension enable --py gmaps')


# In[21]:


gmaps.configure(api_key='AIzaSyDAj29IyNJeRJ3ht8omoyoVzte49xBShNQ')


# In[22]:


df_gmaps = covid19_confirmed.iloc[:,[2,3,-1]]   #only selected columns
df_gmaps


# In[23]:


#replace -1 int the data with 1
df_gmaps[df_gmaps.columns[-1]] = df_gmaps.iloc[:,[-1]].replace(-1,1)


# In[24]:


locations = df_gmaps[['Lat','Long']]
values = df_gmaps.iloc[:,-1]


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
world_gmap = gmaps.figure()
world_gmap.add_layer(gmaps.heatmap_layer(locations, values, max_intensity=700, point_radius=10))
world_gmap


# In[ ]:





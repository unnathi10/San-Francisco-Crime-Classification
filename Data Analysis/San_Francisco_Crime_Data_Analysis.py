
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv("train.csv")
train_data.head()


# In[2]:


top_crimes = train_data.Category.value_counts()[:10]
plt.figure(figsize=(12, 8))
pos = np.arange(len(top_crimes))
plt.barh(pos, top_crimes.values, color='purple');
plt.yticks(pos, top_crimes.index);


# In[3]:


plt.show()


# In[4]:


top_crimes


# In[5]:


top_addresses = train_data.Address.value_counts()[:15]
plt.figure(figsize=(12, 8))

pos = np.arange(len(top_addresses))
plt.bar(pos, top_addresses.values)
plt.xticks(pos, top_addresses.index, rotation = 70)
plt.title('Top 15 Locations with the most crime')
plt.xlabel('Location')
plt.ylabel('Number of Crimes')
plt.show()


# In[7]:


top_days = train_data.DayOfWeek.value_counts()
plt.figure(figsize=(12, 8))

pos = np.arange(len(top_days))
plt.bar(pos, top_days.values)
plt.xticks(pos, top_days.index, rotation = 70)
plt.title('Number of Crimes Occurring each day')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Crimes')
plt.show()


# In[8]:


result = pd.read_csv("Results.csv")
result.head()


# In[10]:


train_data['Hour'] = train_data['Dates'].apply(lambda x:int(x.split(' ')[1].split(':')[0]))
train_data.head()


# In[49]:


hours = train_data.groupby('Hour').size()
hours


# In[51]:


plt.plot(hours.values, 'ro-', color = 'indigo')

plt.xticks(hours.index)
plt.title('Crime Occurence By Hour')
plt.ylabel ('Number of Crimes')
plt.xlabel ('Hour')
plt.show()


# In[4]:


import matplotlib.image as mpimg
california_img=mpimg.imread('sanfrancisco.png')
train_data.plot(kind="scatter", x="Y", y="X", cmap=plt.get_cmap("jet"),colorbar=False, alpha=0.4)
                      
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05],alpha=0.5)
plt.ylabel("Y", fontsize=14)
plt.xlabel("X", fontsize=14)
plt.tick_params(colors='w')


plt.legend(fontsize=16)
plt.show()


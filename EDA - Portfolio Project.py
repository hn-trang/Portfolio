#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model

df = pd.read_csv("melb_data.csv")
df.head()


# In[2]:


#Check data types of all attributes and null values
df.info()


# In[3]:


#Statistical summary of data
df.describe()


# In[4]:


#Check for duplication
df.duplicated().sum()


# In[ ]:


#Check for all observations of attributes


# In[5]:


df['Suburb'].unique()


# In[6]:


df['CouncilArea'].unique()


# In[7]:


df['Regionname'].unique()


# In[ ]:


#Visualization of the attributes and observations count


# In[10]:


sns.set(rc = {'figure.figsize':(15,8)})
sns.countplot(df['CouncilArea']).unique()


# In[ ]:


#Check for null values


# In[11]:


df.isnull().sum()


# In[12]:


#Drop columns with high number of null values and irrelevant column
df = df.drop(['BuildingArea','YearBuilt','CouncilArea','Propertycount'],axis=1)


# In[13]:


#Fill missing values with the values of next observations
df = df.fillna(method='bfill', axis=0).fillna(0)


# In[14]:


df = df[df['Landsize'] != 0]
df


# In[15]:


sns.set(rc = {'figure.figsize':(15,8)})
sns.countplot(df['Regionname']).unique()


# In[ ]:


#Check for outliers with boxplots


# In[16]:


plt.boxplot([df['Rooms'],df['Bedroom2'],df['Bathroom']])


# In[17]:


plt.boxplot(df['Price'])


# In[18]:


plt.boxplot(df['Landsize'])


# In[19]:


del_landsize = df.index[df['Landsize'] > 9000].tolist()
del_landsize


# In[20]:


df = df.drop(labels = [687,
 2084,
 2487,
 3750,
 3942,
 4706,
 5194,
 5584,
 5592,
 5694,
 7778,
 8241,
 8379,
 8828,
 9223,
 10045,
 10488,
 10504,
 10819,
 11020,
 11371,
 11526,
 12163,
 12340,
 12504,
 12594,
 12723,
 13245,
 13389], axis=0)
df.shape


# In[21]:


df.index[df['Bedroom2'] > 10].tolist()


# In[22]:


df = df.drop(labels = [7404], axis=0)
df.shape


# In[ ]:


#Check for distribution and skewness of data


# In[23]:


df.skew(), df.kurt()


# In[24]:


y = df['Price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)


# In[25]:


sns.distplot(df.skew(),color='blue',axlabel ='Skewness')


# In[26]:


plt.figure(figsize = (12,8))
sns.distplot(df.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
plt.show()


# In[27]:


target = np.log(df['Price'])
target.skew()
plt.hist(target,color='blue')


# In[ ]:


#Check for correlation between attributes


# In[28]:


numeric_features = df.select_dtypes(include=[np.number])
correlation = numeric_features.corr()
print(correlation['Price'].sort_values(ascending = False),'\n')


# In[29]:


f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)


# In[30]:


k= 11
cols = correlation.nlargest(k,'Price')['Price'].index
print(cols)
cm = np.corrcoef(df[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


# In[31]:


g = sns.PairGrid(df)
g.map(sns.scatterplot)


# In[32]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

new_df = df[['Suburb','Type','Method','SellerG']].copy()
non_concat_df = df[['Rooms','Price','Distance','Bedroom2','Bathroom','Car','Landsize']].copy()

import category_encoders as ce
encoder = ce.OrdinalEncoder(cols = ['Suburb','Type','Method','SellerG'])
result = encoder.fit_transform(df)
result.head()


# In[51]:


data = df[['Rooms','Car','Landsize','Distance','Price']]

df = data.dropna()


# In[52]:


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
data_scaled= scaler.fit_transform(df)

data_s = pd.DataFrame(data_scaled).describe()
data_s


# In[53]:


from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=2,init='k-means++')
kmeans.fit(data_scaled)


# In[54]:


kmeans.inertia_


# In[55]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init = 'k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
    
frame = pd.DataFrame({'Cluster': range(1,20), 'SSE':SSE})
plt.figure(figsize = (12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[56]:


clusters = [5,6,7,8,9]
pred = []

for cluster in clusters:
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init = 'k-means++')
    kmeans.fit(data_scaled)
    pred.append(kmeans.predict(data_scaled))


# In[57]:


frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred[0]


# In[58]:


frame['cluster'].unique()


# In[59]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,8))
ax = Axes3D(fig)
for i in frame['cluster'].unique():
    ax.scatter(frame[frame.cluster==i][3], frame[frame.cluster==i][2], frame[frame.cluster==i][4])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Landsize')
    ax.set_zlabel('Price')  


# In[ ]:


#Houses in closer distance to city centre are more varried in prices as well as landsize. Purple cluster shows houses in further distance that are less in price and landsize.


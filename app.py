import streamlit as st

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')






def main():

    # global df
    # global raw_data

    df = None
    raw_data =None

    st.write('# Sales Prediction In Store')

    activites = ['EDA', 'PLOT']
    
    choice = st.sidebar.selectbox('Select Activies', activites)
    
    if choice == 'EDA':
        st.subheader('Explotary Data Analysis')
        
        raw_data = st.file_uploader('Upload Dataset', type=['csv', 'txt'])
       
        if raw_data is not None:

           df = pd.read_csv(raw_data)


           st.write(""" 
                ## Raw Data Without Data Cleaning
             """)

           st.dataframe(df.head())



        if st.checkbox('Show Row and Colomn'):
            st.write(str(df.shape[0]) + "  rows " + str(df.shape[1]) + " colunms")
            
        if st.checkbox('Shows Colunms'):
            all_columns = df.columns.to_list()
            st.write(all_columns)


        if st.checkbox('Shows particular Colunms'):
            
           par_columns = st.multiselect('Select Columns', df.columns.to_list())
           
           st.dataframe(df[par_columns])


        if st.checkbox('Shows Summary of Data'):
            
            st.write(df.describe())

        if st.checkbox('Show Null values Counts in Each Columns'):
            st.write(df.isnull().sum())

        st.write('## check for categorical attributes')
        if st.checkbox('check for categorical attributes'):
            cat_col = []
            for x in df.dtypes.index:
                if df.dtypes[x] == 'object':
                    cat_col.append(x)
            st.write(cat_col)


    if choice == 'PLOT':
        st.subheader('Ploting of Data Analysis')

        data = st.file_uploader('Upload Dataset', type=['csv', 'txt'])

        if data is not None:
            df = pd.read_csv(data)
        

        
            st.write("Data")

            st.dataframe(df.head())
        
            if st.checkbox('Chart for Item Weight'):
    
                st.write("## Chart for Item Weight")
                st.write(sns.distplot(df['Item_Weight']))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Chart for Item Visibilty'):
    
                st.write("## Chart for Item Visibilty")
                st.write(sns.distplot(df['Item_Visibility']))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Chart for Item MRP'):
    
                st.write("## Chart for Item MRP")
                st.write(sns.distplot(df['Item_MRP']))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Amount of Fat in Product '):
    
                st.write("## Amount of Fat in Product ")
                df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
                st.write(sns.countplot(df["Item_Fat_Content"]))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Chart for each Product'):
    
                st.write("## Chart for each Product ")
                l = list(df['Item_Type'].unique())
                chart = sns.countplot(df["Item_Type"])
                chart.set_xticklabels(labels=l, rotation=90)
                st.write(chart)
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Sales based on Estblishment Year'):
    
                st.write("## Sales based on Estblishment Year ")
               
                st.write(sns.countplot(df['Outlet_Establishment_Year']))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Sales based on Store Size'):
    
                st.write("## Sales based on Store Size ")
               
                st.write(sns.countplot(df['Outlet_Size']))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();
            if st.checkbox('Coorelation Matrix for Sales'):

                st.write("## Coorelation Matrix for Sales ")
                corr = df.corr()
                
                st.write(sns.heatmap(corr, annot=True, cmap='coolwarm'))
                st.set_option('deprecation.showPyplotGlobalUse', False)


                st.pyplot();


    if choice == 'Model Building':
        st.subheader('Bulding Data Model') 



if __name__ == '__main__':
    main()





# # In[10]:


# #to find unique element in each colun

# df.apply(lambda x : len(x.unique()))


# # In[11]:


# # check for null values
# df.isnull().sum()


# # In[12]:



# # check for categorical attributes
# cat_col = []
# for x in df.dtypes.index:
#     if df.dtypes[x] == 'object':
#         cat_col.append(x)
# cat_col


# # In[13]:


# cat_col.remove('Item_Identifier')
# cat_col.remove('Outlet_Identifier')
# cat_col


# # In[14]:



# # print the categorical columns
# for col in cat_col:
#     print(col)
#     print(df[col].value_counts())
#     print()


# # In[15]:


# # fill the missing values
# item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
# item_weight_mean


# # In[16]:


# miss_bool = df['Item_Weight'].isnull()
# miss_bool


# # In[17]:


# for i, item in enumerate(df['Item_Identifier']):
#     if miss_bool[i]:
#         if item in item_weight_mean:
#             df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
#         else:
#             df['Item_Weight'][i] = np.mean(df['Item_Weight'])


# # In[18]:


# df['Item_Weight'].isnull().sum()


# # In[20]:


# outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
# outlet_size_mode


# # In[24]:


# miss_bool = df['Outlet_Size'].isnull()
# df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# # In[25]:


# df['Outlet_Size'].isnull().sum()


# # In[26]:


# sum(df['Item_Visibility']==0)


# # In[27]:


# # replace zeros with mean
# df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)


# # In[28]:


# sum(df['Item_Visibility']==0)


# # In[29]:


# # combine item fat content
# df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
# df['Item_Fat_Content'].value_counts()


# # In[31]:


# df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
# df['New_Item_Type']


# # In[32]:


# df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
# df['New_Item_Type'].value_counts()


# # In[33]:


# df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
# df['Item_Fat_Content'].value_counts()


# # In[34]:



# # create small values for establishment year
# df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']


# # In[35]:


# df['Outlet_Years']


# # In[36]:


# df.head()


# # In[37]:


# sns.distplot(df['Item_Weight'])


# # In[38]:


# sns.distplot(df['Item_Visibility'])


# # In[39]:


# sns.distplot(df['Item_MRP'])


# # In[40]:


# sns.distplot(df['Item_Outlet_Sales'])


# # In[41]:


# sns.countplot(df["Item_Fat_Content"])


# # In[42]:



# # plt.figure(figsize=(15,5))
# l = list(df['Item_Type'].unique())
# chart = sns.countplot(df["Item_Type"])
# chart.set_xticklabels(labels=l, rotation=90)


# # In[43]:


# sns.countplot(df['Outlet_Establishment_Year'])


# # In[45]:


# sns.countplot(df['Outlet_Location_Type'])


# # In[46]:


# sns.countplot(df['Outlet_Type'])


# # In[47]:


# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')


# # In[ ]:





# # In[ ]:





# # In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:35:24 2021

@author: DELL
"""
import pandas as pd
from ast import literal_eval
import numpy as np
from tkinter import *
movies_data = pd.read_csv('C:/Users/DELL/Desktop/PROJECTS/Netflix recommendation System/movies_metadata.csv',low_memory=False)
movies_data['genres']=movies_data['genres'].fillna('[]').apply(literal_eval)
movies_data['genres']=movies_data['genres'].apply(lambda x: [i['name'] for i in x ] if isinstance(x, list) else [])
genre_split=movies_data.apply(lambda x:pd.Series(x['genres']),axis=1).stack().reset_index(level=1,drop=True)

genre_split.name='Genre'
#print(type(genre_split))
md=movies_data.drop('genres',axis=1).join(genre_split)
#print(md.head(5))
movies_data['year']=pd.to_datetime(movies_data['release_date'],errors='coerce')


movies_data['year']=movies_data['year'].apply(lambda x: str(x).split('-')[0] if x!=np.nan else np.nan)
links_small=pd.read_csv('C:/Users/DELL/Desktop/PROJECTS/Netflix recommendation System/links_small.csv')
#print(links_small.head(5))
links_small=links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
from pandas.api.types import is_numeric_dtype
#print(is_numeric_dtype(md['id']))
md=md.drop([19730,29503,35587])
md['id']=md['id'].astype('int')
#print(md['id'].head(7))
md_new=md[md['id'].isin(links_small)]

md_new_sample=md_new.sample(frac=0.15,random_state=42)

#print(mdnewsample.head(6))
md_new_sample['tagline']=md_new_sample['tagline'].fillna('')
md_new_sample['description']=md_new_sample['overview']+md_new_sample['tagline']
md_new_sample['description']=md_new_sample['description'].fillna("")
#print( md_new_sample['description'].head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english')
tfidf_matrix=tf.fit_transform(md_new_sample['description'])
#print(md_new_sample.head())
#print(tfidf_matrix)
#print(tfidf_matrix.shape,md_new_sample.shape)
cosin_sim=linear_kernel(tfidf_matrix,tfidf_matrix)

#print(md_new_sample.index)
#print(cosin_sim.shape)
pd.Series(md_new_sample.index,index=md_new_sample['title'])

md_new_sample=md_new_sample.reset_index()
#print(md_new_sample.head(6))
titles=md_new_sample['title']
indices=pd.Series(md_new_sample.index,index=md_new_sample['title'])

#print(titles)
#print(indices)
def recommend ():
    txt.delete(0.0, 'end')
    title =ent.get()
   
    idx=indices[title]
  
    sim_scores=list(enumerate(cosin_sim[idx]))
   
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    
    sim_scores=sim_scores[1:31]
    
    
    movie_indices=[i[0] for i in sim_scores]
    
#    l=titles.iloc[movie_indices]
    l=[]
    for i in movie_indices:
        l.append(titles.iloc[movie_indices])
        
    
    for x in range(11):
        t="\n"
        txt.insert(0.0, l[x])
        
        txt.insert(0.0, t)

#print(recommend('Avatar').head(10))
root=Tk()
root.geometry("420x300")
l1=Label(root,text="Enter Movie Name:")
l2=Label(root,text="Top 10 Suggestions for you:")
ent=Entry(root)
l1.grid(row=0)
l2.grid(row=2)
ent.grid(row=0,column=1)
txt=Text(root,width=50,height=13,wrap=WORD)
txt.grid(row=3,columnspan=2, sticky=W)

btn=Button(root,text="Search",bg="purple",fg="white",command=recommend)
btn.grid(row=1,columnspan=2)
root.mainloop()

    







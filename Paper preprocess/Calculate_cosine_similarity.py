from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd
import pickle

# Load the doc2vec model trained through Doc2vec.py.
doc2vec_model= Doc2Vec.load("./doc2vec_document_20211020.model")

# Seed papers are examples of papers you want to selectively collect. 
# In our case, the seed paper is co2 electrochemical reduction reaction paper.
seed_paper = ['Wiley_CO2RR_01444',
                'Wiley_CO2RR_01434',
                'Elsevier_CO2RR_00006',
                'RSC_CO2RR_00294',
                'RSC_CO2RR_01342',
                'Nature_CO2RR_00013',
                'RSC_CO2RR_00845',
                'ACS_CO2RR_00336',
                'Elsevier_CO2RR_00707',
                'Wiley_CO2RR_01181',
                'Wiley_CO2RR_00462',
                'Elsevier_CO2RR_00256',
                'Elsevier_CO2RR_00186',
                'ACS_CO2RR_00210',
                'ACS_CO2RR_00339',
                'ACS_CO2RR_00308',
                'ACS_CO2RR_00201',
                'ACS_CO2RR_00281',
                'Elsevier_CO2RR_00267',
                'Nature_CO2RR_00034']

labels = np.asarray(doc2vec_model.dv.index_to_key)

df = pd.DataFrame(columns=['Name', 'Cosine_similarity'])

cosine_max = 0
distance_min = 100

# After obtaining the cosine similarity between a random paper and seed papers, the max value among the 20 cosine similarities is selected as the final value.
for i, label in enumerate(labels) :
    cosine_max = 0
    distance_min = 100

    for seed in seed_paper :
        label_dv = doc2vec_model.dv[str(label)]
    
        seed_dv = doc2vec_model.dv[str(seed)]
        # cosine = cosine_similarity(seed_dv, label_dv)
        cosine = np.dot(seed_dv, label_dv) /(np.linalg.norm(seed_dv)*np.linalg.norm(label_dv))
        distance = np.linalg.norm((seed_dv - label_dv))

        distance_min = distance if distance_min > distance else distance_min
        # print(seed_dv, label_dv)
        # print(cosine, distance)

        # cosine_max = cosine if cosine_max < cosine else cosine_max
        if cosine_max >= cosine :
            cosine_max = cosine_max
        
        else :
            cosine_max = cosine


    print(str(label)+' : '+str(cosine_max))
    
    df = df.append(pd.DataFrame([[label, cosine_max, distance_min]], columns=['Name', 'Cosine_similarity', 'distance']), ignore_index=True)
    
df.to_excel('cosine_doc_20211020.xlsx', index=False)


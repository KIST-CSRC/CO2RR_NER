from gensim.models.ldamodel import LdaModel
from chemdataextractor.doc.text import Text
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
import os
import time  
import pickle

from nltk.corpus import stopwords 
from gensim import corpora
import pandas as pd
import csv
import pyLDAvis
import pyLDAvis.gensim_models

from gensim.models.coherencemodel import CoherenceModel 
import matplotlib.pyplot as plt
import warnings

import nltk
nltk.download('averaged_perceptron_tagger')  

class Lda:
    def __init__(self):
        pass
    
    # Lowercase words that are not chemical formulas using chemdataextractor
    def normalize(self, no_chem_word):
        no_chem_word = no_chem_word.lower()
        return no_chem_word   

    # Tokenize the corpus using chemdataextractor's tokenizer
    # Create a list of preprocessed tokens
    def preprocess(self, corpus):
        abstract_corpus = Text(corpus)
        chemicals = [chemiclas.text for chemiclas in abstract_corpus.cems]
        sentence_token_result = []

        for i, sentence_token in enumerate(abstract_corpus.tokens):
            
            sentence_token_list = []

            for word in sentence_token:
                word = word.text  

                if word not in chemicals : 
                    normalized_word = self.normalize(word)
                    sentence_token_list.append(normalized_word)

                else :
                    sentence_token_list.append(word)
            
            sentence_token_result.append(sentence_token_list)

        print("finished preprocess")
        return sentence_token_result

    # Tag words by identifying parts of speech in sentences
    def pos_tagging(self, corpus):

        POS_list = []

        for sentence in self.preprocess(corpus):
            POS_list.append(nltk.pos_tag(sentence))
            
        # print(POS_list)
        print("finished pos tagging")
        return POS_list

    # Lemmatization is a technique that combines multiple forms of a word into a single form. 
    # For example, when lemmatization is performed on the three words ‘am’, ‘are’, and ‘is’, the result is ‘be’.
    # Unlike stemming, which is mechanically determined by empirical laws without considering the semantic unit of a word, Lemmatization considers the semantic unit of a word and is performed through morphological analysis, so word-level analysis is more accurate.
    def lemmatization(self, corpus):

        lemma_list = []
        lemmatizer = nltk.stem.WordNetLemmatizer()

        for sentence in self.pos_tagging(corpus):

            for token, POS in sentence :
                
                # When using the Lemmatize function, you can tell what part of speech the corresponding token is. If parts of speech are not provided, incorrect results may be returned.
                # The parts of speech input to the Lemmatize function are only verbs, adjectives, nouns, and adverbs. It is input as v, a, n, r, respectively.
                # Among parts of speech returned by nltk.pos_tag, adjectives start with J, so you need to change them to a before lemmatize.
                func_j2a = lambda x : x if x != 'j' else 'a'

                if POS[0] in ['V', 'J', 'N', 'R'] :
                    pos_contraction = [(token, func_j2a(POS.lower()[0]))]
                    # print(pos_contraction)
                    for token, POS in pos_contraction :

                        lemma_list.append(lemmatizer.lemmatize(token, POS))

                else :
                    lemma_list.append(token)
        
        # print(*lemma_list, sep='\n')
        print("finished lemmatizaion")
        return lemma_list
    
    def stemming(self, corpus):
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        words = self.stopwords(corpus)
        
        result_token = [stemmer.stem(w) for w in words]
        
        print(result_token)
        return result_token

    # Remove stopwords, remove words of length one
    def stopwords(self, corpus):
        lemma_list = self.lemmatization(corpus)
        # print(self.lemmatization(corpus))
        stop_words = set(stopwords.words('english')) 
        result_token = []

        for w in lemma_list:
            if w not in stop_words:
                if len(w) != 1 :
                    
                    result_token.append(w) 
        # print(result_token)
        print("finished stopwords")
        return result_token

    def lda_preprocess(self, path_dir):
            
        file_list = os.listdir(path_dir)
        token_doc = []

        for file in file_list:
            
            with open(path_dir+'/'+file, 'r', encoding='UTF8') as f:
                corpus = f.read()

            result_token = self.stopwords(corpus)
            token_doc.append(result_token)
            print('tokenization finished' +'\t'+ str(file[:-4]))
        
        # Replace each word with (word id, number of occurrences)    
        dictionary = corpora.Dictionary(token_doc)

        # Remove words with a frequency of 20 or less
        dictionary.filter_extremes(no_below=20)
        print(len(dictionary))
        corpus = [dictionary.doc2bow(word) for word in token_doc]

        return corpus, dictionary, token_doc

    # Run function
    def run(self, path_dir) :
        corpus, dictionary, token_doc= self.lda_preprocess(path_dir)

        ldamodel = LdaModel(corpus, num_topics = 15, id2word=dictionary,iterations=500, passes=40, alpha = 'auto', eta = 'auto')
        topics = ldamodel.print_topics()
        file_list = os.listdir(path_dir)

        
        vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
        pyLDAvis.save_html(vis, 'LDA_15_20211108.html')
        
        # print(*topics, sep='\n')

        # for topic in topics:
        with open("./topic_15_20211108.csv",'a', encoding="UTF-8") as f:
                csv.writer(f).writerows(topics)

        topic_table = pd.DataFrame()

        for i, topic_list in enumerate(ldamodel[corpus]):
            doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
            doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
            topic_table = topic_table.rename(index={i:file_list[i]})
            
            for j, (topic_num, prop_topic) in enumerate(doc): 
                if j == 0:  
                    topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                    
                else:
                    continue

        for i, topic_list in enumerate(ldamodel[corpus]):   
            topic_table = topic_table.rename(index={i:file_list[i][:-4]})
        topic_table.insert(0,"Name",file_list,True)

        print(topic_table)
        topic_table.to_excel('topic_table_15_20211108.xlsx', index=False)
    
    # The two functions below are evaluation methods to properly determine the number of topics, which is a hyperparameter of LDA.
    
    # Coherence measures the coherence of a topic. The better the topic model is, the more semantically similar words are gathered in one topic.
    # Therefore, the higher the coherence value, the higher the semantic consistency.
    def coherence_perplexity(self, path_dir): 
        corpus, dictionary, token_doc = self.lda_preprocess(path_dir)
        coherence_values = []
        perplexity = []

        for i in range(2,40):            
            ldamodel = LdaModel(corpus, num_topics = i , id2word=dictionary, passes=40, iterations=500, alpha = 'auto', eta = 'auto')
            coherence_model_lda =CoherenceModel(model=ldamodel, texts = token_doc, dictionary = dictionary)
            coherence_lda = coherence_model_lda.get_coherence()
            coherence_values.append(coherence_lda)
            print('finished coherence topic('+str(i)+')')
            perplexity.append(ldamodel.log_perplexity(corpus))
            print('finished perplexitiestopic('+str(i)+')')

        x = range(2,40)
        plt.plot(x, coherence_values)
        plt.xlabel("number of topics")
        plt.ylabel("coherence score")
        plt.show()

        x = range(2,40)
        plt.plot(x, perplexity)
        plt.xlabel("number of topics")
        plt.ylabel("perplexity score")
        plt.show()
    
    # Perpelxity is said to be written as confusion in the dictionary. That is, how well a particular probabilistic model predicts values ​​that are actually observed. 
    # If the Perlexity value is small, it can be seen that the topic model reflects the document well.
    def perplexity(self, path_dir):
        corpus, dictionary, token_doc = self.lda_preprocess(path_dir)
        perplexity_values = []

        for i in range(2, 20):
            ldamodel = LdaModel(corpus, num_topics = i , id2word=dictionary, passes=40)
            perplexity_values.append(ldamodel.log_perplexity(corpus))

        x = range(2,20)
        plt.plot(x, perplexity_values)
        plt.xlabel("number of topics")
        plt.ylabel("perplexity score")
        plt.show()
    
if __name__ == '__main__':

    print(time.strftime('%c', time.localtime(time.time())))
    
    path_dir = 'D:/corpus/abstract/after_wiley/spacing_error_x/abstract_20211020/total_corpus'

    lda_obj = Lda()
    lda_obj.run(path_dir)

    print(time.strftime('%c', time.localtime(time.time())))

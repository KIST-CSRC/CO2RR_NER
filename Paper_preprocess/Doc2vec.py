from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from chemdataextractor.doc.text import Text
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
import os
import time  
import numpy as np

class DocumentVector:
    def __init__(self, vector_size=200, alpha=0.025, min_alpha=0.00025, min_count=1, workers=4, dm=1, epochs=10):

        self.vector_size = vector_size
        self.alpha = alpha
        self.min_count = min_count
        self.min_alpha = min_alpha
        self.workers = workers
        self.dm = dm
        self.epochs = epochs

    # Lowercase words that are not chemical formulas using chemdataextractor
    def normalize(self, no_chem_word):
        no_chem_word = no_chem_word.lower()
        return no_chem_word   
    
    # Tokenize the corpus using chemdataextractor's tokenizer
    def tokenize(self, corpus):
        corpus = Text(corpus)

        total_token_list_loc = []
        total_token_loc = []
        total_token = []

        for word_token in corpus.tokens:
            
            for word in word_token:
                total_token_loc.append((word.start, word.end))
                total_token.append(word.text)


        for pair in zip(total_token_loc, total_token):
            total_token_list_loc.append(list(pair))

        return total_token

    # Create a list of preprocessed tokens using the above two functions
    def preprocess(self, corpus):

        total_token = self.tokenize(corpus)

        cwt = ChemWordTokenizer()

        for i in range(0, len(total_token)):
            chemicals_token_list = []
            normalized_word_list = []

            entity_word = Text(total_token[i])
        
            if entity_word.cems != [] :
                    for chemicals in entity_word.cems :
                        chemicals = chemicals.text 
                        chemicals_token_list.append(chemicals)
                        
            if str(entity_word) not in chemicals_token_list : 
                entity_word = entity_word.text
                normalized_word = entity_word.lower()
                normalized_word_list.append(normalized_word)
                total_token[i] = normalized_word

        return total_token

    # Train a doc2vec model and save the trained model
    def doc2vec(self, tag_data) :

        model = Doc2Vec(vector_size = self.vector_size ,
                        alpha = self.alpha,
                        min_count = self.min_count,
                        min_alpha = self.min_alpha,
                        workers = self.workers,
                        dm = self.dm)

        model.build_vocab(tag_data)


        model.train (tag_data,
                    total_examples=model.corpus_count,
                    epochs=self.epochs)
        model.save("doc2vec_document_20211020.model")
        
        return model

    # Run function
    def run(self, path_dir) :
        tag_data = []
        file_list = os.listdir(path_dir)

        for file in file_list:
            
            with open(path_dir+'/'+file, 'r', encoding='UTF8') as f:
                corpus = f.read()

            total_token = self.preprocess(corpus)

            tag_data.append(TaggedDocument(words=total_token, tags=[str(file[:-4])]))
            
            print('tokenization finished' +'\t'+ str(file[:-4]))


        model = self.doc2vec(tag_data)


if __name__ == '__main__':
    print(time.strftime('%c', time.localtime(time.time())))

    path_dir = 'D:/corpus/corpus_20211013'

    documentvector = DocumentVector()
    documentvector.run(path_dir)

    print(time.strftime('%c', time.localtime(time.time())))                

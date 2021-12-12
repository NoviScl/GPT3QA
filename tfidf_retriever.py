from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
import os

from typing import Union, Dict
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAVE_DIR = 'retriever_caches'
MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'


class TfidfGuesser:
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self):
        """
        Initializes data structures that will be useful later.
        """        
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", stop_words='english', max_df=0.5)
        self.tfidf = None
        self.questions = None
        self.answers = None

    def train(self, training_data):
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """
        
        questions = [x["question"] for x in training_data]
        self.answers = [x["answer"][0] for x in training_data] # only take the first answer
        self.questions = questions

        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(questions)
        self.tfidf = self.tfidf_vectorizer.transform(questions)

    def save(self, QA_PATH):
        with open(os.path.join(SAVE_DIR, QA_PATH, MODEL_PATH), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(os.path.join(SAVE_DIR, QA_PATH, INDEX_PATH), 'wb') as f:
            pickle.dump(self.tfidf, f)

        with open(os.path.join(SAVE_DIR, QA_PATH, QN_PATH), 'wb') as f:
            pickle.dump(self.questions, f)

        with open(os.path.join(SAVE_DIR, QA_PATH, ANS_PATH), 'wb') as f:
            pickle.dump(self.answers, f)
        
        
        

    def guess(self, question, max_n_guesses=4):
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        question -- Raw text of the question
        max_n_guesses -- How many top guesses to return
        """
        top_questions = []
        top_answers = []
        
        question_tfidf = self.tfidf_vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        cos = cosine_similarities[0]
        indices = cos.argsort()[::-1]

        for i in range(max_n_guesses):
            idx = indices[i]
            top_questions.append(self.questions[idx])
            top_answers.append(self.answers[idx])
        
        return top_questions, top_answers

    def load(self, QA_PATH):
        with open(os.path.join(SAVE_DIR, QA_PATH, MODEL_PATH), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open(os.path.join(SAVE_DIR, QA_PATH, INDEX_PATH), 'rb') as f:
            self.tfidf = pickle.load(f)
        
        with open(os.path.join(SAVE_DIR, QA_PATH, QN_PATH), 'rb') as f:
            self.questions = pickle.load(f)

        with open(os.path.join(SAVE_DIR, QA_PATH, ANS_PATH), 'rb') as f:
            self.answers = pickle.load(f)



if __name__ == "__main__":
    
    all_qa = ["BeerQA", "HotpotQA", "QANTA", "TimeQA", "CSQA2", "NQ", "StrategyQA", "TriviaQA"]
    for QA_data in all_qa:
        os.makedirs(os.path.join(SAVE_DIR, QA_data), exist_ok = True)
        
        guesstrain = os.path.join("DiverseQA", QA_data + "_train.json")
        guesstest = os.path.join("DiverseQA", QA_data + "_test.json")

        print("Loading %s" % guesstrain)
        with open(guesstrain, "r") as f:
            train = json.load(f)

        tfidf_guesser = TfidfGuesser()
        tfidf_guesser.train(train)
        tfidf_guesser.save(QA_data)    


    # # ## load and test
    # print("Loading %s" % guesstest)
    # with open(guesstest, "r") as f:
    #     test = json.load(f)

    # tfidf_guesser = TfidfGuesser()
    # tfidf_guesser.load(QA_data)
    
    # for qn in test:
    #     question = qn["question"]
    #     print (question)
    #     top_q, top_a = tfidf_guesser.guess(question = question, max_n_guesses = 4)
    #     print (top_q)
    #     print (top_a)
    #     print ()

   
    

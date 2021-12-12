## GPT-3 For QA

The original version of the system is in `gpt3_expo.py`. The upgraded system, which we call **GPR: GPT-3 with Prompt Retrieval**, is implemented in `gpr_qa.py`. The current system supports a wide range of QA datasets. We include eight QA datasets in the directory `DiverseQA` which can be directly loaded and evaluated.

We use a simple TF-IDF retrieval system to retrieve the most relevant prompts for each test question and use them to construct the prompt. The retriever (i.e., TfidfVectorizer) and processed passage indices are stored in the directory `retriever_caches` and the retriever itself is implemented in `tfidf_retriever.py`. 
Note that the retriever is implemented and tested with `sklearn 1.0`. 

Based on a human evaluation on 100 randomly sampled test questions from QANTA, GPR achieves an EM score of 91, while the GPT-3 baseline achieves 84. 

We also included the model predictions of GPT-3 and GPR on the eight test sets in DiverseQA in the directory `predictions`. We plan to perform some manual analysis and present some insights in the report. 







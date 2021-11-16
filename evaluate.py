import string
import re 
import numpy as np
import json

def get_exact_match(answers1, answers2):
    if type(answers1)==list:
        if len(answers1)==0:
            return 0
        return np.max([get_exact_match(a, answers2) for a in answers1])
    if type(answers2)==list:
        if len(answers2)==0:
            return 0
        return np.max([get_exact_match(answers1, a) for a in answers2])
    return (normalize_answer(answers1) == normalize_answer(answers2))


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1(answers, predictions, is_equal=get_exact_match, return_p_and_r=False):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    assert len(answers)>0 and len(predictions)>0, (answers, predictions)
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if return_p_and_r:
        if a+b==0:
            return 0., 0., 0.
        return 2*a*b/(a+b), float(a), float(b)
    if a+b==0:
        return 0.
    return 2*a*b/(a+b)



if __name__ == "__main__":
    with open("test_100.json", 'r') as f:
        test = json.load(f)
    answers = []
    for eg in test:
        ans = eg["answer"].split('[')[0].strip()
        ans = ans.split('(')[0].strip()
        answers.append(ans)
    
    with open('predictions_last_sent.json', 'r') as f:
        predictions = json.load(f)
    
    EM = 0
    for i in range(len(answers)):
        em = get_exact_match(answers[i], predictions[i])
        EM += em
        # if em == 0:
        #     print ("question: ", test[i]["text"])
        #     print ("gold: ", answers[i])
        #     print ("pred: ", predictions[i])
        #     print ('\n')
    print (EM)

    # print (get_f1(answers, predictions))
        
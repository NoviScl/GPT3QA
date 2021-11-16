import os
import openai
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from evaluate import * 

engine = 'davinci'
openai.api_key = '' ## fill in your key and adjust the GPT-3 version that you wanna use
openai.Engine.retrieve(engine)

## randomly sampled train examples to serve as prompts
with open('train_10.json', 'r') as f:
  train = json.load(f)

## randomly sampled test set for evaluation
with open('test_100.json', 'r') as f:
  test_100 = json.load(f)

def quizbowl_prompt(num_demo=4):
	prompt = ''
	for i in range(num_demo):
		prompt += train[i]["text"] + '\n'
		answer = train[i]["answer"]
		answer = answer.split('[')[0].strip()
		prompt += "The answer is " + answer + "\n\n"
	return prompt


def predict(question, threshold=100):
	prompt = quizbowl_prompt(4)
	input_prompt = prompt + question + '\n'
	input_prompt += "The answer is"
	aa = openai.Completion.create(
		engine=engine,
		prompt=input_prompt,
		temperature=0.0,
		max_tokens=8,
		top_p=1.0,
		frequency_penalty=0.0,
		presence_penalty=0.0,
		logprobs=1,
		stop=['\n']
	)
	ans = aa['choices'][0]['text']
	confidence = sum(aa['choices'][0]["logprobs"]["token_logprobs"])
	## some simple cleaning
	ans = ans.split()
	ans_lst = []
	for a in ans:
		ans_lst.append(a)
		if a[-1] == '.':
			break
	ans = ' '.join(ans)

	# print ("question: ", question)
	# print ("prediction: ", ans)
	# print ("confidence: ", confidence)
	# print ("length: ", len(question.split()))

	# if (confidence > -10.0 and len(question.split()) > 100) or (confidence > -6.0 and question.split() > 60):
	# 	return ans, confidence
	# else:
	# 	return None
	if len(question.split()) > threshold:
		return ans
	else:
		return None


def quizbowl_test(samples=100, threshold=100):
	answers = []
	test = test_100[ : samples]
	EM = 0
	total_len = 0
	buzz_len = 0
	buzzed = 0
	for eg in test:
		total_len += len(eg["text"].split())
		text_orig = sent_tokenize(eg["text"])
		for s in range(len(text_orig)):
			question = ' '.join(text_orig[:s+1])
			pred = predict(question, threshold=threshold)
			
			if pred is not None:
				buzz_len += len(question.split())
				buzzed += 1

				## clean up gold answer
				ans = eg["answer"].split('[')[0].strip()
				ans = ans.split('(')[0].strip()
				# print ("answer: ", eg["answer"])
				# print ("\n")

				# print ("pred: ", pred)
				# print ("gold: ", ans)
				# print ()

				em = get_exact_match(ans, pred)
				EM += em

				break
	print ("EM score: ", EM)
	print ("AVG question len: ", total_len / 100)
	print ("buzzed: ", buzzed)
	print ("AVG len of buzz: ", buzz_len / buzzed)
	print ("\n")
	return answers



if __name__ == "__main__":
	for threshold in [90, 100, 110]:
		print ("buzz threshold: ", threshold)
		answers = quizbowl_test(100, threshold=threshold)



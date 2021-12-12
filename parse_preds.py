import json

with open("predictions/QANTA_test_PromptRetrieve_preds.json", "r") as f:
    data = json.load(f)

for dp in data:
    print ("pred: ", dp["prediction"])
    print ("answer: ", dp["gold_answer"])
    print ()


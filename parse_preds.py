import json

with open("predictions/QANTA_test_GPT_preds.json", "r") as f:
    data1 = json.load(f)

with open("predictions/QANTA_test_PromptRetrieve_preds.json", "r") as f:
    data2 = json.load(f)


for i in range(len(data1)):
    dp = data1[i]
    dp2 = data2[i]
    if dp["prediction"] != dp2["prediction"]:
        print ("question: ", dp["question"])
        print ("pred  GPT: ", dp["prediction"])
        print ("pred GPR: ", dp2["prediction"])
        print ("answer: ", dp["gold_answer"])
        print ()


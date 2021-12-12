import json 
import numpy as np
np.random.seed(2021)

# with open("/home/sichenglei/quizbowl/data/TimeQA/dataset/human_annotated_test.json", "r") as f:
#     data = json.load(f)

# qdata = []
# for d in data:
#     for qd in d["questions"]:
#         nqd = {}
#         nqd["question"] = qd[0]
#         nqd["answer"] = []
#         for dd in qd[1]:
#             nqd["answer"].append(dd["answer"])
#         qdata.append(nqd)

# data = qdata

# print (qdata)
# print (len(qdata))

with open('/home/sichenglei/quizbowl/data/HotpotQA/hotpot_train_v1.1.json', 'r') as f:
    data = json.load(f)
data = data[ : 1000]
# data = data["questions"]

sampled = np.random.choice(data, 100)
final = []
q_len = 0
a_len = 0
for d in sampled:
    new_d = {}
    new_d["question"] = d["question"]
    q_len += len(new_d["question"].split())
    # new_d["answer"] = [str(d["answer"])]
    new_d["answer"] = [d["answer"]]
    # new_d["answer"] = [d["answer"].split('[')[0].strip()]
    a_len += len(new_d["answer"][0].split())
    final.append(new_d)

print ("num: ", len(final))
print ("q len: ", q_len / len(final))
print ("a len: ", a_len / len(final))

with open("/home/sichenglei/quizbowl/DiverseQA/HotpotQA_test.json", "w") as f:
    json.dump(final, f)

# with open('/home/sichenglei/quizbowl/data/CSQA2/dataset/CSQA2_train.json', 'r') as f:
#     data = list(f)

# data = [json.loads(l) for l in data]

# sampled = np.random.choice(data, 1000)
# final = []
# q_len = 0
# a_len = 0
# for d in sampled:
#     new_d = {}
#     new_d["question"] = d["question"]
#     q_len += len(new_d["question"].split())
#     new_d["answer"] = [d["answer"]]
#     a_len += len(new_d["answer"][0].split())
#     final.append(new_d)

# print ("num: ", len(final))
# print ("q len: ", q_len / len(final))
# print ("a len: ", a_len / len(final))

# with open("/home/sichenglei/quizbowl/DiverseQA/CSQA2_train.json", "w") as f:
#     json.dump(final, f)
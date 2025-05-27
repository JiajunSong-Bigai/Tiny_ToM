from utils import i_to_c, c_to_i
import json
import random

dataset = json.load(open("data/dataset.json", "r"))


def test():
    data = dataset["data"]

    example = data[random.sample(range(len(data)), 1)[0]]
    print(example)
    tokens = c_to_i(example["events"] + " " + example["question"])
    print(tokens)
    answer_token = c_to_i(example["answer"])

    print(i_to_c(tokens))
    print(i_to_c(answer_token))


test()

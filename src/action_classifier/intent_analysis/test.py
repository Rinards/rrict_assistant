import json
# import process_data
from .model2 import Net as Net2
from sklearn.metrics.pairwise import cosine_similarity
from .label_manager import load_labels
import numpy as np

import torch
import sys


def test(fnc, model_labels):
    class_num = len(model_labels)

    assert class_num > 0

    # We a assume all classes have the same number of samples
    test_sample_size = len(test_data[model_labels[0]])

    total = test_sample_size * class_num
    total_correct = 0
    for intent in test_data:
        sentences = test_data[intent]
        labels = [intent] * test_sample_size

        pred = fnc(sentences)
        correct = pred.count(intent)
        total_correct += correct
        print(f"number of matched for {intent} : {correct}")

    accuracy = total_correct * 100 / total
    print(f"{total}/{total_correct} : accuracy of {accuracy}")
    return total, total_correct


if __name__ == '__main__':
    labels_path = "./labels.json"
    test_data_path = "datasets/test_data.json"
    model_path = "./MODEL"

    # Parse command line arguments
    if len(sys.argv) < 3: # No dataset
        print("Provide a dataset and labels by runing the program with: ")
        print("./train path/to/dataset.json path/to/labels.json")
        exit(1)
    else:
        test_data_path = sys.argv[1]
        labels_path = sys.argv[2]

        if len(sys.argv) > 3:
            model_path = sys.argv[3]

    model_labels = load_labels(labels_path)
    class_num = len(model_labels)

    test_data = None
    with open(test_data_path) as f:
        test_data = json.load(f)

    model2 = Net2(class_num)

    print("Running fine-tuned model")
    model2.load_state_dict(torch.load(model_path))
    model2.eval()

    def run_model2(sentences):
        labels = []
        Y = model2(sentences)
        for intent_cls in Y:
            labels.append(model_labels[intent_cls.argmax()])
        return labels

    test(run_model2, model_labels)


from .model2 import Net, run_model
import torch
from .label_manager import load_labels
import sys


if __name__ == '__main__':
    labels_path = "./labels.json"
    model_path = "./MODEL"

    # Parse command line arguments
    if len(sys.argv) < 2: # No dataset
        print("Provide labels by runing the program with: ")
        print("./iteractive path/to/labels.json")
        exit(1)
    else:
        labels_path = sys.argv[1]

        if len(sys.argv) > 2:
            model_path = sys.argv[2]

    model_labels = load_labels(labels_path)
    class_num = len(model_labels)

    model = Net.pretrained(class_num, model_path)

    while True:
        user_input = input("You: ")
        if user_input.upper() == "/QUIT":
            break

        label = run_model(model, [user_input], model_labels)[0]
        print(f"LABEL: {label}")


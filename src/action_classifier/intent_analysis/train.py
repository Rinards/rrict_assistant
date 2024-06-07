import torch
from .model2 import Net, train, tokenizer
from .data_loader import create_data_loader, IntentRecognitionDataset
from .process_data import json_to_pandas
from .label_manager import load_labels

import json
import sys


if __name__ == '__main__':
    train_data_path = None
    labels_path = None
    model_path = "./MODEL"

    # Parse command line arguments
    if len(sys.argv) < 3: # No dataset
        print(sys.argv)
        print("Provide a dataset and labels by runing the program with: ")
        print("./train path/to/dataset.json path/to/labels.json")
        exit(1)
    else:
        train_data_path = sys.argv[1]
        labels_path = sys.argv[2]

        if len(sys.argv) > 3:
            model_path = sys.argv[3]

    model_labels = load_labels(labels_path);
    class_num = len(model_labels)
    print(model_labels);

    json_data = None;
    with open(train_data_path, 'r') as json_file:
        json_data = json.load(json_file)

    df = json_to_pandas(json_data, model_labels);
    df = df.sample(frac=1) # shuffle

    model = Net(class_num)
    dataloader = create_data_loader(df, tokenizer, 128, 16)

    train(model, dataloader)
    torch.save(model.state_dict(), model_path)

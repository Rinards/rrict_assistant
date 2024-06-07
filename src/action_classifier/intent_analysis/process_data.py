import json
import pandas as pd
from typing import Any

# TODO: add this in a label file and import that
def json_to_pandas(json_dict: dict[any], model_labels: list[str]) -> pd.DataFrame:
    data = [
        (model_labels.index(intent), sentence)
        for intent, sentences in json_dict.items() for sentence in sentences
    ]
    df = pd.DataFrame(data, columns=['intent', 'prompt']);
    return df

if __name__ == '__main__':
    json_data = None;
    with open('datasets/test_data.json', 'r') as json_file:
        json_data = json.load(json_file)

    df = json_to_pandas(json_data);
    print(df)

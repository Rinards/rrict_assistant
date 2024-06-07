import json;

def load_labels(label_path: str) -> None:
    model_labels = []
    with open(label_path, "r") as labels_json:
        model_labels = json.loads(labels_json.read())

    return model_labels

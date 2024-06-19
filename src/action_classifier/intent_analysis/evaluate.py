import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from intent_analysis.model2 import Net, run_model
from intent_analysis.label_manager import load_labels

# Load the test data
with open('../../datasets/test_data.json', 'r') as f:
    test_data = json.load(f)

# Prepare the test dataframe by flattening the structure
test_texts = []
test_labels = []
for label, texts in test_data.items():
    test_texts.extend(texts)
    test_labels.extend([label] * len(texts))

test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

# Instantiate the model
labels_path = "../../labels.json"
model_path = "./MODEL"
model_labels = load_labels(labels_path)
class_num = len(model_labels)

model = Net.pretrained(class_num, model_path)

# Make predictions
y_pred = run_model(model, test_df['text'].tolist(), model_labels)

# Evaluate the model
unique_classes = sorted(list(set(test_df['label']) | set(y_pred)))
conf_matrix = confusion_matrix(test_df['label'], y_pred, labels=unique_classes)
class_report = classification_report(test_df['label'], y_pred, labels=unique_classes, target_names=unique_classes)
accuracy = accuracy_score(test_df['label'], y_pred)

# Print evaluation results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("\nAccuracy:", accuracy)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from seqeval.metrics import classification_report,accuracy_score
from eval import valid
from data import test_data_loader


print("\n Evaluation Result: \n")
labels, predictions = valid (test_data_loader,)

print("Accuracy:")
print(accuracy_score(labels, predictions))
print(classification_report([labels], [predictions]))
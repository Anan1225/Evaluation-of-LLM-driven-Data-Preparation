import matplotlib.pyplot as plt
import numpy as np

models = ["GPT-5.2", "GPT-4o", "GeminiFast", "GeminiPro", "Llama4"]
x = np.arange(len(models))
width = 0.6

TP = np.array([77, 73, 81, 81, 6])
TN = np.array([0, 25, 25, 23, 1])
FP = np.array([27, 2, 2, 4, 26])
FN = np.array([5, 9, 1, 1, 76])

plt.figure(figsize=(6, 3))

plt.bar(x, TP, width, label="TP")
plt.bar(x, TN, width, bottom=TP, label="TN")
plt.bar(x, FP, width, bottom=TP + TN, label="FP")
plt.bar(x, FN, width, bottom=TP + TN + FP, label="FN")

plt.xticks(x, models)
plt.ylabel("Count")
plt.title("Confusion Matrix Components by Model")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("tptn.png", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt

models = ["GPT-5.2", "GPT-4o", "GeminiFast", "GeminiPro", "Llama4"]

precision = [74.04, 97.33, 97.59, 95.30, 18.75]
recall    = [93.90, 89.02, 98.78, 98.78, 7.31]
f1        = [82.79, 92.99, 98.18, 97.01, 10.53]
accuracy  = [70.64, 89.91, 97.25, 95.41, 6.42]

plt.figure(figsize=(6,3))

plt.plot(models, precision, marker="o", label="Precision")
plt.plot(models, recall,    marker="s", label="Recall")
plt.plot(models, f1,        marker="^", label="F1")
plt.plot(models, accuracy,  marker="d", label="Accuracy")

plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.title("Evaluation Metrics by Model")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("tptn_metrics.png", bbox_inches="tight")
plt.show()


#%%
# Loading library python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# %%# %%
# Read dataset
df_net = pd.read_csv('citrus.csv')
df_net.head()


# %%
# Describe data
df_net.describe()

#%%
df_net.value_counts()


# %%
X = df_net.drop(columns="name")
y = LabelEncoder().fit_transform(df_net["name"])  # Mengubah "orange" jadi 1 dan "grapefruit" jadi 0 (misalnya)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model = GaussianNB()
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Akurasi:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# %%
# Misalnya conf_matrix adalah hasil dari confusion_matrix(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# Visualisasi confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Grapefruit', 'Orange'],
            yticklabels=['Grapefruit', 'Orange'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Naive Bayes (Citrus Dataset)')
plt.tight_layout()
plt.show()
# %%
# Data baru (diameter, weight, red, green, blue)
data_dummy_grapefruit = np.array([
    [4.2, 98.0, 168, 75, 12],
    [4.4, 102.0, 172, 73, 10],
    [4.3, 100.0, 170, 70, 9],
    [4.1, 97.5, 165, 78, 11],
    [4.5, 101.0, 174, 74, 13],
    [4.3, 99.0, 169, 76, 10],
    [4.2, 100.5, 171, 72, 8],
    [4.4, 103.0, 175, 71, 7],
    [4.3, 98.5, 167, 74, 12],
    [4.1, 96.0, 164, 77, 9]
])

# Prediksi kelas
predicted_class = model.predict(data_dummy_grapefruit)

# Prediksi kelas dengan model
predictions = model.predict(data_dummy_grapefruit)

# Konversi ke label asli
predicted_labels = label_encoder.inverse_transform(predictions)

for i, label in enumerate(predicted_labels, 1):
    print(f"Data {i}: Prediksi = {label}")
# %%

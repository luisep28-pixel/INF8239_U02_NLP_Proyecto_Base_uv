import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Configuración de reproducibilidad y entorno
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("reports/figures", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# 1. Carga y Preprocesamiento de Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 2. Arquitecturas: Baseline Denso (DNN) vs. CNN
class DenseBaseline(nn.Module):
    def __init__(self):
        super(DenseBaseline, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.flatten(x))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# 3. Función de Entrenamiento y Evaluación
def train_and_evaluate(model, name, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    elapsed_time = time.time() - start_time

    # Evaluación en Test Set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f"[{name}] Accuracy: {acc:.4f} | F1 Macro: {f1:.4f} | Tiempo: {elapsed_time:.2f}s")
    return acc, f1, elapsed_time, all_targets, all_preds


# 4. Ejecución del Experimento Comparativo
dnn_acc, dnn_f1, dnn_time, targets, dnn_preds = train_and_evaluate(DenseBaseline(), "Baseline DNN", epochs=5)
cnn_acc, cnn_f1, cnn_time, targets, cnn_preds = train_and_evaluate(ConvNet(), "CNN (ConvNet)", epochs=5)

# Guardar Resultados en CSV
df_metrics = pd.DataFrame({
    'Modelo': ['Baseline (Red Densa DNN)', 'Red Convolucional (CNN)'],
    'Accuracy Global': [dnn_acc, cnn_acc],
    'F1 Macro': [dnn_f1, cnn_f1],
    'Tiempo Entrenamiento (s)': [dnn_time, cnn_time],
    'Huella CO2 Estimada (gCO2eq)': [dnn_time * 0.004, cnn_time * 0.004]  # Estimación Green AI
})
df_metrics.to_csv("reports/metrics_lab07_models.csv", index=False)

# ----------------------------------------------------
# 5. Generación de Gráficos de Evidencia
# ----------------------------------------------------

# Gráfico 1: Comparativa de Rendimiento
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(2)
width = 0.35

ax.bar(x - width / 2, [dnn_acc, cnn_acc], width, label='Accuracy', color='#2b5c8f')
ax.bar(x + width / 2, [dnn_f1, cnn_f1], width, label='F1 Macro', color='#e67e22')

ax.set_ylabel('Puntuación (0 - 1.0)', fontweight='bold')
ax.set_title('U02.LAB07: Comparativa de Rendimiento (DNN vs CNN)', fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Baseline (Red Densa DNN)', 'Red Convolucional (CNN)'], fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("reports/figures/lab07_comparativa_rendimiento.png", dpi=300)
plt.close()

# Gráfico 2: Matriz de Confusión Normalizada de la CNN
cm = confusion_matrix(targets, cnn_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax, cbar=True)
ax.set_title('Matriz de Confusión Normalizada (CNN - Fashion-MNIST Test Set)', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Clase Predicha', fontweight='bold')
ax.set_ylabel('Clase Real', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("reports/figures/lab07_matriz_confusion_cnn.png", dpi=300)
plt.close()

# Gráfico 3: Green AI y Eficiencia Computacional
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.bar(x - width / 2, [dnn_time, cnn_time], width, color='#1f77b4', alpha=0.85, label='Tiempo (s)')
ax1.set_ylabel('Tiempo de Entrenamiento (segundos)', color='#1f77b4', fontweight='bold')

ax2 = ax1.twinx()
ax2.bar(x + width / 2, [dnn_time * 0.004, cnn_time * 0.004], width, color='#d62728', alpha=0.85, label='CO2 (g)')
ax2.set_ylabel('Huella Carbono Estimada (gCO2eq)', color='#d62728', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(['Baseline (Red Densa DNN)', 'Red Convolucional (CNN)'], fontweight='bold')
ax1.set_title('Green AI: Costo Computacional y Huella Ecológica', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("reports/figures/lab07_green_ai_costo.png", dpi=300)
plt.close()

print("\n¡LAB07 completado con éxito! Archivos e imágenes creados en 'reports/'.")
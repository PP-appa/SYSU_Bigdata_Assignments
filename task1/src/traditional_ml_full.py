import os
import json
import numpy as np
import torchvision
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def main():
    # 1. Load full CIFAR-10 dataset
    print("Loading full CIFAR-10 dataset for traditional ML models...")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Flatten the images and normalize to [0, 1]
    X_train = train_dataset.data.reshape(50000, -1).astype('float32') / 255.0
    y_train = np.array(train_dataset.targets)
    X_test = test_dataset.data.reshape(10000, -1).astype('float32') / 255.0
    y_test = np.array(test_dataset.targets)

    # 2. PCA Dimensionality Reduction
    # Reduces the 3072 features to 150 components to make training feasible for traditional models.
    print("Executing PCA (Principal Component Analysis)...")
    pca = PCA(n_components=150, whiten=True, random_state=42)
    start_time = time.time()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA completed in {time.time() - start_time:.2f} seconds.")

    # 3. Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    start_time = time.time()
    rf_model.fit(X_train_pca, y_train)
    rf_train_time = time.time() - start_time

    y_pred_rf = rf_model.predict(X_test_pca)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest - Accuracy: {acc_rf * 100:.2f}%, Training Time: {rf_train_time:.2f}s")

    # 4. Support Vector Machine (SVM)
    print("Training SVM Classifier (RBF kernel)...")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale')
    start_time = time.time()
    svm_model.fit(X_train_pca, y_train)
    svm_train_time = time.time() - start_time

    y_pred_svm = svm_model.predict(X_test_pca)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM - Accuracy: {acc_svm * 100:.2f}%, Training Time: {svm_train_time:.2f}s")

    # 5. Export results to JSON
    result_file = './output/model_results.json'
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    results = {}
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass

    results['Random_Forest'] = {'accuracy': float(f"{acc_rf*100:.2f}"), 'train_time': float(f"{rf_train_time:.2f}")}
    results['SVM'] = {'accuracy': float(f"{acc_svm*100:.2f}"), 'train_time': float(f"{svm_train_time:.2f}")}

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    main()

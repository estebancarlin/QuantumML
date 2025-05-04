import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def plot_decision_regions(X, y, classifier, resolution=0.004):
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=cl)
    plt.legend()


def iris_classical_svm():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    svm = SVC(kernel='rbf', gamma=1.0, C=10.0)
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X_train_std, y_train, svm)
    plt.title("SVM on Iris (train set)")
    plt.show()

    print("Test Accuracy:", svm.score(X_test_std, y_test))
    return X_train, X_test, y_train, y_test


def iris_quantum_svc(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_t = scaler.transform(X_test)

    feature_map = PauliFeatureMap(feature_dimension=2, reps=2, paulis=['Z', 'ZZ'])
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=quantum_kernel)

    qsvc.fit(X_scaled, y_train)
    y_pred = qsvc.predict(X_scaled_t)

    accuracy = accuracy_score(y_test, y_pred)
    print("Quantum SVC Accuracy:", accuracy)


def cancer_classical_svm():
    cancer = datasets.load_breast_cancer()
    X = cancer.data[:, :2]  # simplify for 2D plot
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    svm = SVC(kernel='rbf', gamma=1.0, C=10.0)
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X_train_std, y_train, svm)
    plt.title("SVM on Breast Cancer (train set)")
    plt.show()

    print("Test Accuracy:", svm.score(X_test_std, y_test))
    return X_train, X_test, y_train, y_test


def cancer_quantum_svc(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_t = scaler.transform(X_test)

    feature_map = PauliFeatureMap(feature_dimension=2, reps=2, paulis=['Z', 'ZZ'])
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=quantum_kernel)

    qsvc.fit(X_scaled, y_train)
    y_pred = qsvc.predict(X_scaled_t)

    accuracy = accuracy_score(y_test, y_pred)
    print("Quantum SVC Accuracy:", accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['iris_classical', 'iris_quantum', 'cancer_classical', 'cancer_quantum', 'all'], default='all')
    args = parser.parse_args()

    if args.experiment == 'iris_classical':
        iris_classical_svm()
    elif args.experiment == 'iris_quantum':
        X_train, X_test, y_train, y_test = iris_classical_svm()
        iris_quantum_svc(X_train, X_test, y_train, y_test)
    elif args.experiment == 'cancer_classical':
        cancer_classical_svm()
    elif args.experiment == 'cancer_quantum':
        X_train, X_test, y_train, y_test = cancer_classical_svm()
        cancer_quantum_svc(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = iris_classical_svm()
        iris_quantum_svc(X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = cancer_classical_svm()
        cancer_quantum_svc(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()

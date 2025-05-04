
# Quantum Machine Learning Comparative Study

This repository presents an educational exploration and comparison of **classical** and **quantum** machine learning algorithms. 
This project was developed within the semester **Sciences de l'information et société numérique (SISN)** (Information Science and Digital Society) at **Ecole Centrale Marseille**, as part of a semester-long group assignment.
The members of this project were : Eliott Saltre, CLément Perrin, Kris Joubi and Esteban Carlin.

The project is divided into two main components:

1. **Quantum vs Classical Perceptron Models** — implemented in `QML_cleaned.py`
2. **Quantum SVM vs Classical SVM** on real-world datasets — implemented in `SVMvsQSVM_cleaned.py`

This work was inspired and built upon excellent resources from:
- [awesome-quantum-machine-learning](https://github.com/krishnakumarsekar/awesome-quantum-machine-learning)
- [Quantum Perceptron Models by mroget](https://github.com/mroget/Quantum-perceptron-models)

---

## Files Overview

### `QML_cleaned.py`

This file explores five different perceptron variants trained and evaluated on the **Iris dataset** and a synthetic "hard" dataset.

Algorithms included:
- `Online Perceptron` (Classical)
- `Version Space Perceptron` (Classical)
- `Quantum Version Space Perceptron`
- `Quantum Online Perceptron`
- `Hybrid Quantum Perceptron`

The quantum models use simulated **Grover’s Search Algorithm** to accelerate convergence during training. The implementation includes:
- Analytical and simulated Grover probability computation
- Margin calculation using `LinearSVC`
- Grover-based oracle optimizations and probabilistic amplification

### `SVMvsQSVM_cleaned.py`

This script compares the performance of:
- `SVM (Support Vector Machine)` with RBF kernel (classical)
- `QSVM (Quantum SVM)` using Qiskit Machine Learning’s `QSVC`

Datasets used:
- **Iris** (2D subset for visualization)
- **Breast Cancer** (2D projection)

It includes:
- Standardization (`StandardScaler`)
- MinMax scaling for quantum kernel input
- Visual decision region plots
- CLI interface to run different configurations via `--experiment` flag

---

## How to Use

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the SVM experiment script:

```bash
python SVMvsQSVM_cleaned.py --experiment iris_classical
python SVMvsQSVM_cleaned.py --experiment iris_quantum
python SVMvsQSVM_cleaned.py --experiment cancer_classical
python SVMvsQSVM_cleaned.py --experiment cancer_quantum
```

Or execute all:

```bash
python SVMvsQSVM_cleaned.py --experiment all
```

---

## Results Summary

| Model                         | Dataset      | Key Advantage                               |
|-------------------------------|--------------|---------------------------------------------|
| Classical Perceptron          | Iris/Hard    | Simplicity and robustness                   |
| Quantum Perceptrons           | Iris/Hard    | Fewer iterations for high-margin cases      |
| Classical SVM                 | Iris/Cancer  | High accuracy, stable                       |
| Quantum SVM (QSVC)            | Iris/Cancer  | Competitive accuracy with quantum potential |

---

## Key Concepts

- **Grover's Algorithm**: Used for probabilistic search amplification in quantum perceptrons.
- **FidelityQuantumKernel**: Used in QSVC for measuring similarity in a quantum feature space.
- **Perceptron Variants**: Classical and quantum adaptations of linear classifiers.
- **Margin-Based Learning**: Model initialization and capacity tied to dataset margins.

---

##  Credits

This project is inspired and supported by the following excellent open-source repositories:

- 🔗 [krishnakumarsekar/awesome-quantum-machine-learning](https://github.com/krishnakumarsekar/awesome-quantum-machine-learning)
- 🔗 [mroget/Quantum-perceptron-models](https://github.com/mroget/Quantum-perceptron-models)

Special thanks to the creators and contributors of Qiskit and Scikit-learn.

---

## License

This project is intended for **educational and research use only**. Please respect the licenses of upstream repositories referenced in the Credits.

---

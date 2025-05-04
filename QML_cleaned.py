# === Data Manipulation ===
import numpy as np
import pandas as pd

# === Mathematical Tools ===
import math
from math import pi

# === Plotting ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Classical Machine Learning ===
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split

# === Plot Configuration ===
sns.set()
sns.set_context("poster")   # Larger fonts for presentation
sns.set_style("ticks")      # Clean tick style

def get_grover_operator(classical_oracle):
    """
    Build the Grover diffusion operator from a classical oracle.

    Args:
        classical_oracle (list of bool or int): Marks (1 or True) indicate targets.

    Returns:
        psi (np.ndarray): Initial uniform quantum state vector.
        G (np.ndarray): Grover operator matrix.
    """
    N = len(classical_oracle)  # Size of the search space

    # Initial uniform superposition state
    psi = np.array([1.0 / np.sqrt(N)] * N)

    # Grover diffusion operator (reflection over the mean)
    U = np.array([[i] for i in psi])
    U = 2.0 * np.dot(U, U.conj().T) - np.eye(N)

    # Oracle: flips the sign of marked elements
    R = np.eye(N)
    for i in range(N):
        if classical_oracle[i]:
            R[i][i] = -1.0

    # Grover operator: G = U * R
    G = np.dot(U, R)

    return psi, G

# === Example call with oracle marking first state ===
psi, G = get_grover_operator([1, 0, 0, 0])
print(np.round(G, 2))
print(psi)

def success_proba_grover(classical_oracle, m):
    """
    Compute the probability of successfully measuring a marked state in Grover's algorithm.

    Args:
        classical_oracle (list of bool or int): Target(s) to be marked.
        m (int): Number of Grover iterations.

    Returns:
        steps (np.ndarray): Step indices.
        p (np.ndarray): Success probability at each step.
    """
    def success_proba(psi):
        return sum(abs(psi[i]) ** 2 for i in range(len(psi)) if classical_oracle[i])

    N = len(classical_oracle)
    psi, G = get_grover_operator(classical_oracle)

    steps = [0]
    p = [success_proba(psi)]

    for step in range(1, m + 1):
        psi = np.dot(G, psi)
        steps.append(step)
        p.append(success_proba(psi))

    return np.array(steps), np.array(p)

# === Visualization example for N = 8 ===
oracle = [1] + [0] * 7  # One marked element at index 0
x, y = success_proba_grover(oracle, 20)

# Theoretical success probability
theta = np.arcsin(np.sqrt(sum(oracle) / len(oracle)))
def theoretical_success_prob(t):
    return np.sin(2 * theta * t + theta) ** 2

# Plotting
plt.figure(figsize = (12, 7))
plt.plot(x, y, 'o--', label = "Grover algorithm (simulated)")
plt.plot(np.linspace(x[0], x[-1], 100), (np.linspace(x[0], x[-1], 100)), '--', label = "Theoretical success rate")

plt.xlabel("Steps")
plt.ylabel("Probability of Success")
plt.legend()
plt.title("Success Probability of Grover's Algorithm Over Time")
plt.grid(True)
plt.show()

def grover(classical_oracle, m):
    """
    Run Grover's algorithm for `m` iterations and return the measured result.

    Args:
        classical_oracle (list): Boolean or int list marking the targets.
        m (int): Number of Grover iterations to apply.

    Returns:
        int: Index of the measured element after applying Grover's algorithm.
    """
    N = len(classical_oracle)
    psi, G = get_grover_operator(classical_oracle)
    
    # Apply Grover operator m times
    psi = np.dot(np.linalg.matrix_power(G, m), psi)
    
    # Measure: sample according to the final state's probability amplitudes
    probabilities = np.abs(psi) ** 2
    result = np.random.choice(N, 1, p = probabilities)
    
    return result[0]

# === Run Grover 1000 times and estimate probability distribution ===
oracle = [1] + [0] * 7  # Only one marked item
results = [grover(oracle, 2) for _ in range(1000)]

# Estimate empirical probability
proportion = [results.count(i) / len(results) for i in range(len(oracle))]

# Plot the histogram (log scale)
plt.figure(figsize = (8, 5))
plt.bar(range(len(oracle)), proportion)
plt.yscale("log")
plt.xlabel("Element Index")
plt.ylabel("Probability of Measurement (log scale)")
plt.title("Distribution of Measurement Outcomes (1000 runs)")
plt.grid(True)
plt.show()

def grover_opti(classical_oracle, m):
    """
    Simulate Grover's result using analytical formula without matrix multiplications.

    Args:
        classical_oracle (list): Target-marked oracle.
        m (int): Number of Grover iterations.

    Returns:
        int: Simulated measurement result.
    """
    N = len(classical_oracle)
    M = sum(classical_oracle)
    theta = np.arcsin(np.sqrt(M / N))
    
    # Theoretical probability of success after m steps
    success_prob = lambda t: np.sin(2 * theta * t + theta) ** 2

    # Construct full probability distribution across all N elements
    probabilities = [
        success_prob(m) / M if classical_oracle[i] else (1 - success_prob(m)) / (N - M)
        for i in range(N)
    ]
    
    return np.random.choice(N, 1, p = probabilities)[0]

def classical_search(classical_oracle):
    """
    Classic linear search through the oracle.

    Args:
        classical_oracle (list): Target-marked list.

    Returns:
        tuple: (index found, steps used)
    """
    for i, marked in enumerate(classical_oracle):
        if marked:
            return i, i + 1
    return 0, len(classical_oracle) + 1

def quantum_search(classical_oracle, nb_ampli = 10, opti = False):
    """
    Quantum search with optional Grover optimization and amplification.

    Args:
        classical_oracle (list): Target-marked oracle.
        nb_ampli (int): Max attempts if measurement fails.
        opti (bool): Whether to use analytical version (grover_opti) or full simulation (grover).

    Returns:
        tuple: (element found, number of Grover steps used)
    """
    N = len(classical_oracle)
    
    # Theoretical upper bound on iterations
    M_max = math.ceil(1.0 / math.sin(2 * np.arcsin(np.sqrt(1 / N))))
    steps_used = 0
    result = 0
    
    for _ in range(nb_ampli):
        m = np.random.randint(0, M_max + 1)  # Random number of iterations
        steps_used += m
        
        result = grover_opti(classical_oracle, m) if opti else grover(classical_oracle, m)
        
        if classical_oracle[result]:
            break  # Success
        result = 0  # Reset if failed

    return result, steps_used

class Perceptron_Online:
    
    def __init__(self,max_iter = 10000,shuffle = True,quantum = False,nb_ampli = 10,opti = False):
        self.max_iter = max_iter # Number maximal of iteration
        self.shuffle = shuffle # If we shuffle the training dataset between each correction
        self.coef_ = np.array([0.]) # Default hyperplane
        self.n_iter_ = 0 # Number of iteration (include the number of steps used for the search)
        self.n_correction_ = 0 # Number of correction
        self.quantum = quantum # If we use the quantum search
        self.nb_ampli = nb_ampli # The amplification parameter for the quantum search
        self.opti = opti # If we use the opti Grover
    
    def fit(self,X,y):
        """
        Training function of the model.
        Args:
X -> Points
            y -> Classes
        Output :
            The number of iteration.
        The coefficient of the model will be updated during the learning.
        """
        b = True
        self.coef_ = np.array([0.]*len(X[0])) # Initialisation of the coef (can be removed if you want to train successively)
        # Copy of the entry (for the shuffle step)
        X_ = np.array([[j for j in i] for i in X])
        y_ = np.array([1 if i==1 else -1 for i in y]) # Security to ensure that the classes are {-1,1} and not {0,1}.
        
        nb = 0
        while b:
            nb+= 1
            if nb > self.max_iter:
                break
            b = False
            
            oracle = [int(y_[i]*X_[i,:].dot(self.coef_)<=0) for i in range(len(y_))] # Oracle for the search
            
            # The right search (according to the options) is used.
            m,steps = classical_search(oracle) if not self.quantum else quantum_search(oracle,nb_ampli = self.nb_ampli,opti = self.opti)
            self.n_iter_+= steps # We add the number of steps to the model

            if y_[m]*X_[m,:].dot(self.coef_)<=0: # If the search is successful we correct
                self.coef_ = self.coef_ + y_[m]*X_[m,:]
                b = True
                self.n_correction_+= 1
                nb+= 1
            
            
            if self.shuffle: # Shufffle
                l = list(range(len(X)))
                np.random.shuffle(l)
                X_ = np.array([X[i] for i in l])
                y_ = np.array([1 if y[i]==1 else -1 for i in l])
        return self.n_iter_
    
    def predict(self,X):
        """
        Entry : Points
        Output : Classification
        """
        return np.array([1 if x.dot(self.coef_)>0 else -1 for x in X])

class Perceptron_Space:
    def __init__(self,separators,nb_ampli = 10,quantum = False,opti = False):
        self.separators = separators # Set of separators
        np.random.shuffle(self.separators)
        self.selected = 0 # The selected separator
        self.n_iter_ = 0 # Number of steps
        self.coef_ = self.separators[self.selected] # Default hyperplane
        self.quantum = quantum # If we use the quantum search
        self.nb_ampli = nb_ampli # The amplification parameter for the quantum search
        self.opti = opti # If we use the opti Grover
    
    def fit(self,X,y):
        """
        Training function of the model.
        Args:
X -> Points
            y -> Classes
        Output :
            The number of iteration.
        One of the "separators" will be chosen. 
        """
        X_ = np.array([[j for j in i] for i in X])
        y_ = np.array([1 if i==1 else -1 for i in y]) # Security to ensure that the classes are {-1,1} and not {0,1}.
        
        # Oracle over the hyerplanes
        oracle = [int(all([y_[i]*X_[i,:].dot(self.separators[k])>0 for i in range(len(X_))])) for k in range(len(self.separators))]
        
        # Search
        k,steps = classical_search(oracle) if not self.quantum else quantum_search(oracle,nb_ampli = self.nb_ampli,opti = self.opti)
        
        self.n_iter_+= steps*len(X_) # The number of steps is multiplied by the comlexity of the oracle.
        if all([y_[i]*X_[i,:].dot(self.separators[k])>0 for i in range(len(X_))]): # If the search is successful we take the hyperplane.
                self.selected = k
                self.coef_ = self.separators[self.selected]
        
        return self.n_iter_
    
    def predict(self,X):
        """
        Entry : Points
        Output : Classification
        """
        return np.array([1 if x.dot(self.separators[self.selected])>0 else -1 for x in X])

class Perceptron_Hybrid:
    def __init__(self,separators,nb_ampli = 10,quantum = False,opti = False):
        self.separators = separators # Set of separators
        np.random.shuffle(self.separators)
        self.selected = 0 # The selected separator
        self.n_iter_ = 0 # Number of steps
        self.coef_ = self.separators[self.selected] # Default hyperplane
        self.quantum = quantum # If we use the quantum search
        self.nb_ampli = nb_ampli # The amplification parameter for the quantum search
        self.opti = opti # If we use the opti Grover
    
    def fit(self,X,y):
        """
        Training function of the model.
        Args:
X -> Points
            y -> Classes
        Output :
            The number of iteration.
        One of the "separators" will be chosen. 
        """
        X_ = np.array([[j for j in i] for i in X])
        y_ = np.array([1 if i==1 else -1 for i in y]) # Security to ensure that the classes are {-1,1} and not {0,1}.
        
        for k in range(len(self.separators)): # For each separators
            # Oracle over the points
            oracle = [int(y_[i]*X_[i,:].dot(self.separators[k])<=0) for i in range(len(y_))]
            # Search
            m,steps = classical_search(oracle) if not self.quantum else quantum_search(oracle,nb_ampli = self.nb_ampli,opti = self.opti)
        
            self.n_iter_+= steps
        
            if y_[m]*X_[m,:].dot(self.separators[k])>0: # If successful we chose this hyperplane
                    self.selected = k
                    self.coef_ = self.separators[self.selected]
                    break
        
        return self.n_iter_
    
    def predict(self,X):
        """
        Entry : Points
        Output : Classification
        """
        return np.array([1 if x.dot(self.separators[self.selected])>0 else -1 for x in X])

model_name = ["Online perceptron", "Version space perceptron", "Version space quantum perceptron", "Online quantum perceptron", "Hybrid quantum perceptron"]

def init_model(name,X,gamma = 0.01,eps = 0.01):
    """
    This function's purpose is to initalize the models with the right parameters (epsilon, gamma, ...).
    Args:
name -> Name of the model (as listed in model_name)
        X -> The points (the margin, dimension and number of point are used in order to precise the best parameter)
        gamma -> If you want to specify the margin yourself. Useful when the margin is very small and the computation very long.
        eps -> The mistake parameter when chosing the number of hyperplanes and amplifying the quantum search's probability of success
    Output :
        A model with the coresponding parameters.
    Note : The version space quantum algorithm uses the optimmized version of Grover by default because it's the slowest.
    It's here if you want to change this.
    """
    len_data = len(X[0])
    if name==model_name[0]: # Classical perceptron
        return Perceptron_Online(max_iter = int(len(X)/gamma**2),shuffle = False)

    if name==model_name[1]: # Version space classical perceptron
        nb_hyperplanes = math.ceil(math.log(eps)/math.log(1-math.sqrt(2/pi)*gamma))
        separators = np.random.multivariate_normal([0]*len_data,np.eye(len_data),size = nb_hyperplanes)
        return Perceptron_Space(separators)

    if name==model_name[2]: # Version space quantum perceptron
        nb_hyperplanes = math.ceil(np.log(eps)/np.log(1-np.sqrt(2/pi)*gamma))
        nb_ampli = math.ceil(np.log(eps)/np.log(3/4))
        separators = np.random.multivariate_normal([0]*len_data,np.eye(len_data),size = nb_hyperplanes)
        return Perceptron_Space(separators, nb_ampli = nb_ampli, quantum = True, opti = True) # Optimization here

    if name==model_name[3]: # Online quantum perceptron
        nb_ampli = math.ceil(np.log(eps*gamma**2)/np.log(3/4))
        return Perceptron_Online(max_iter = int(len(X)/gamma**2), nb_ampli = nb_ampli, shuffle = False, quantum = True)

    if name==model_name[4]: # Hybrid quantum perceptron
        nb_hyperplanes = math.ceil(np.log(eps/2)/np.log(1-np.sqrt(2/pi)*gamma))
        nb_ampli = math.ceil(np.log(1-(1-eps/2)**(1/(nb_hyperplanes-1)))/np.log(3/4))
        separators = np.random.multivariate_normal([0]*len_data,np.eye(len_data),size = nb_hyperplanes)
        return Perceptron_Hybrid(separators, nb_ampli = nb_ampli, quantum = True)

def compute_margin(X,y,fit_intercept = False):
    """
    Use SVM of sklearn to compute the margin of a dataset.
    Args:
X -> The points
        y -> The classes
        fit_intercept -> If you want to allow the intercept to be fitted. The model coded here haven't this option.
    Returns:
w -> The best separator / the most centered one / the on that realizes the margin.
        b -> The intercept
        gamma -> The margin
    """
    y = [1 if i==1 else -1 for i in y]
    clf = LinearSVC(fit_intercept = fit_intercept,max_iter = 100000)
    clf.fit(X, y)
    w = clf.coef_[0]
    b = 0
    if fit_intercept:
        b = clf.intercept_[0]
    gamma = min([y[i]*(X[i,:].dot(w)+b)/np.linalg.norm(w) for i in range(len(X))])
    w = w/np.linalg.norm(w)
    return w,b,gamma

def get_iris():
    """
    Load Iris with some restrictions.
    """
    iris = datasets.load_iris()
    X_ = iris.data[:,:2]
    Y_ = iris.target
    X = np.array([X_[i]+np.array([0,3.3]) for i in range(len(X_)) if Y_[i] != 2])
    y = np.array([1 if i==0 else -1 for i in Y_ if i != 2])
    size_max = max([np.sqrt(sum(x**2)) for x in X])
    X = np.array([[i/size_max for i in x] for x in X])
    
    u = sum(X)/len(X)
    X = X-u + np.array([0.013,0.])
    
    return X,y

# Plot of the dataset
X,y = get_iris()
plt.figure(figsize = (6,6))
plt.scatter(X[:,0], X[:,1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')

_,_,gamma = compute_margin(X,y,fit_intercept = False)
print(gamma)

def get_linear_separator(model,X,y):
    def get_lin(u,v):
        """
        Get a linear function from two points.
        """
        a = (u[1]-v[1])/(u[0]-v[0])
        b = u[1]-a*u[0]
        return lambda x:a*x+b

    def get_points(w):
        """
        Get two points from a hyperplane w.
        """
        assert(w[1]!=0)
        # Point (1,_)
        u = [1,0]
        u[1] = -u[0]*w[0]/w[1]
        # Point (-1,_)
        v = [-1,0]
        v[1] = -v[0]*w[0]/w[1]
        return u,v
    
    model.fit(X,y)

    w = model.coef_

    u,v = get_points(w)
    f = get_lin(u,v)
    return f

X,y = get_iris()

_,_,gamma = compute_margin(X,y,fit_intercept = False)
print(gamma)
models = [init_model(i,X,gamma = gamma/10) for i in model_name]
f = []

for i in models:
    f.append(get_linear_separator(i,X,y))

plt.figure(figsize = (10,10))
plt.scatter(X[:,0], X[:,1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
t = np.linspace(min(X[:,0]),max(X[:,0]),1000)
for i in range(len(f)):
    plt.plot(t,f[i](t),'-',label = model_name[i])
plt.legend()

def simul(X,y,eps = 0.01,train_size = 0.66,m = 100,gamma = None):
    """
    This function train each models m times and get the mean of the score and number of steps.
    Args:
X -> The points
        y -> The classes
        eps -> The amplification parameter
        train_size -> ratio of the dataset used for training.
        m -> Number of times the process is repeated befor looking at the mean.
        gamma -> If we want to specify the value of gamma.
    Returns:
Panda Dataframe
        algo -> The model used
        N -> The size of the dataset
        gamma -> The margin of the dataset
        train_size -> The ratio of the dataset used for training.
        score -> The proportion of points of the testing dataset that are correctly classified after training.
        nb_operation -> The number of steps used by the algorithm (include the steps for the search and the cost of the oracle).
    """
    if gamma==None:
        _,_,gamma = compute_margin(X,y,fit_intercept = False)
    len_data = len(X[0])
    ret = []
    
    progress = 0
    print("\r{}%     ".format(100*progress/(m*len(model_name))),end = "")
    for name in model_name[:]:
        score = 0
        nb_iter = 0
        for i in range(m):
            model = init_model(name,X,gamma = gamma)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size)
            while not(-1 in y_train and 1 in y_train and -1 in y_test and 1 in y_test):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size)
            model.fit(X_train,y_train)
            nb_iter += model.n_iter_
            score += np.mean([int(z==y) for z,y in zip(model.predict(X_test),y_test)])
            
            progress+= 1
            print("\r{}%     ".format(100*progress/(m*len(model_name))),end = "")
            
        score/= m
        nb_iter/= m
        ret.append([name,len(X),gamma,train_size,score,nb_iter])
        
    print("\rdone    ")
    return pd.DataFrame(np.array(ret,dtype = object),
                        columns = ["algo","N","gamma","train_size","score","nb_operations"])

X,y = get_iris()
np.random.seed(1) # To ensure reproductibility
res_iris = simul(X,y,m = 10,train_size = 0.1)
res_iris["dataset"] = "Iris"
res_iris

N = 200
X = []
for i in range(0,N):
    X.append([(0)**i if j<i else ((-1)**(i+1) if i==j else 0) for j in range(N)])
y = np.array([1 if i%2==0 else -1 for i in range(N)])
X = np.array(X)

np.random.seed(1) # To ensure reproductibility
res_th = simul(X,y,m = 10,eps = 0.01,train_size = 0.9)
res_th["dataset"] = "Hard"
res_th

plt.figure(figsize = (16,5))

names = ["Classical\nperceptron","Version space\nperceptron",
        "Version space\nquantum perceptron","Online quantum\n perceptron","Hybrid quantum\nperceptron"]

res_iris["nb_operations"]/= res_iris["nb_operations"][0]
res_iris["algo"] = names
res_iris["dataset"] = "Iris"

res_th["nb_operations"]/= res_th["nb_operations"][0]
res_th["algo"] = names
res_th["dataset"] = "Hard"

res_final = pd.concat([res_iris,res_th])
res_final = res_final[res_final["algo"] != names[0]]
res_final = res_final[res_final["algo"] != names[1]]

sns.barplot(x = "algo",y = "nb_operations",hue = "dataset",data = res_final)
plt.xlabel("")
plt.ylabel("number of operations \ncompared to the perceptron")
plt.yscale("log")
plt.legend()
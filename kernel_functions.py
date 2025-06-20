import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import time

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from qiskit.quantum_info import DensityMatrix,partial_trace, entropy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def fidelity_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type):
        
    num_features = train_features.shape[1] #num qubits
    m = train_features.shape[0]  #num data points
        
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=ZZ_reps, entanglement=ent_type)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    t0 = time.time()
    gram_matrix_train = kernel.evaluate(x_vec = train_features)
    t1 = time.time()
    print(t1-t0, 'sec to evaluate kernels')
    mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
    independent_entries = gram_matrix_train[mask]

    
    #SVC train and test
    svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

    svc.fit(gram_matrix_train, train_labels)

    gram_matrix_test = kernel.evaluate(x_vec = test_features, y_vec=train_features)

    score_kernel = svc.score(gram_matrix_test, test_labels)
    t2 = time.time()
    print('tot time: ', t2-t0, 'sec \n')
    print(f"Precomputed kernel classification test score: {score_kernel}\n")

    # Get predicted probabilities
    y_pred_probs = svc.predict(gram_matrix_test)

    # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Compute confusion matrix (normalized)
    labels = np.unique(test_labels)  # Dynamically determine class labels
    confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

    # Print Accuracy Per Class and Report
    print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
    print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))


    return independent_entries, score_kernel, confusion


####################################################################

#PROJECTED KERNELS

def Schatten_2_norm(X):
    X_squared = np.matmul(X.conj().T, X)
    norm = np.sqrt(np.trace(X_squared))
    if(np.imag(norm)>1e-10):
        print("Non-zero imaginary part of norm")
        return('Error')
    return(np.real(norm))

def kernel_value_from_exponent_norms(exponent_norms, gamma = 1):
    x = np.power(exponent_norms,2).sum()
    return(np.exp(-gamma*x))

def kernel_value(x1_id, x2_id, all_partial_traces_matrix, gamma = 1):
    exponent_norms = []
    partial_rhos_1 = all_partial_traces_matrix[x1_id]
    partial_rhos_2 = all_partial_traces_matrix[x2_id]

    for i in range(all_partial_traces_matrix.shape[1]):
        partial_rho_1 = partial_rhos_1[i]
        partial_rho_2 = partial_rhos_2[i]
        exponent_norms.append(Schatten_2_norm(partial_rho_1-partial_rho_2))
    exponent_norms = np.array(exponent_norms)
    
    return(kernel_value_from_exponent_norms(exponent_norms, gamma=gamma))

def projected_kernel(X, ZZ_reps=1, ent_type='linear'):

    N_FEATURES = X.shape[1]
    all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
    qubit_list = [i for i in range(N_FEATURES)]

    for data_point in range(X.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        # fm_bound = fm.bind_parameters(X[data_point])
        fm_bound = fm.assign_parameters(X[data_point])
        qc.append(fm_bound, range(N_FEATURES))
        rho = DensityMatrix.from_instruction(qc)

        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

        for current_qubit in qubit_list:
            list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
            partial_rho=partial_trace(rho,list_to_trace_out).data
            partial_rhos[current_qubit] = partial_rho

        all_partial_traces[data_point] = partial_rhos

    kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

    for row in range(kernel_matrix.shape[0]):
        for column in range(row+1,kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

    kernel_matrix += kernel_matrix.T
    kernel_matrix += np.identity(kernel_matrix.shape[0])

    return all_partial_traces, kernel_matrix

def projected_kernel_test(X, Y, x_partial_traces, ZZ_reps=1, ent_type='linear'):

    N_FEATURES = X.shape[1]

    y_partial_traces = np.empty((Y.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
    qubit_list = [i for i in range(N_FEATURES)]

    for data_point in range(Y.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        # fm_bound = fm.bind_parameters(X[data_point])
        fm_bound = fm.assign_parameters(Y[data_point])
        qc.append(fm_bound, range(N_FEATURES))
        rho = DensityMatrix.from_instruction(qc)

        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

        for current_qubit in qubit_list:
            list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
            partial_rho=partial_trace(rho,list_to_trace_out).data
            partial_rhos[current_qubit] = partial_rho

        y_partial_traces[data_point] = partial_rhos

    all_partial_traces = np.concatenate((x_partial_traces, y_partial_traces), axis=0)

    kernel_matrix = np.zeros((Y.shape[0], X.shape[0]))

    for row in range(kernel_matrix.shape[0]):
        for column in range(kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = X.shape[0] + row, x2_id = column, all_partial_traces_matrix = all_partial_traces)   

    # kernel_matrix += kernel_matrix.T
    # kernel_matrix += np.identity(kernel_matrix.shape[0])
    return(kernel_matrix) 

def projected_kernel_with_entropy(X, ZZ_reps=1, ent_type='linear'):

    N_FEATURES = X.shape[1]
    all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
    qubit_list = [i for i in range(N_FEATURES)]

    all_entropies = np.empty((X.shape[0]))

    for data_point in range(X.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        # fm_bound = fm.bind_parameters(X[data_point])
        fm_bound = fm.assign_parameters(X[data_point])
        qc.append(fm_bound, range(N_FEATURES))
        rho = DensityMatrix.from_instruction(qc)

        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

        for current_qubit in qubit_list:
            list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
            partial_rho=partial_trace(rho,list_to_trace_out).data
            partial_rhos[current_qubit] = partial_rho

        all_partial_traces[data_point] = partial_rhos

        reduced_rho = partial_trace(rho,range(round(N_FEATURES/2)))
        all_entropies[data_point] = entropy(reduced_rho, base=2)

    kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

    for row in range(kernel_matrix.shape[0]):
        for column in range(row+1,kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

    kernel_matrix += kernel_matrix.T
    kernel_matrix += np.identity(kernel_matrix.shape[0])

    entanglement_entropy = np.mean(all_entropies)

    return all_partial_traces, kernel_matrix, entanglement_entropy

def projected_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type, compute_entropy = None, mps_sim = None):

    print(compute_entropy, mps_sim)
    # num_features = train_features.shape[1] #num qubits
    m = train_features.shape[0]  #num data points

    t0 = time.time()
    
    entanglement_entropy = 0 
    
    if compute_entropy:
        all_partial_traces, gram_matrix_train, entanglement_entropy = projected_kernel_with_entropy(train_features, ZZ_reps=ZZ_reps, ent_type=ent_type)
    elif mps_sim:
        all_partial_traces, gram_matrix_train = projected_kernel_mps(train_features, ZZ_reps=ZZ_reps, ent_type=ent_type)
    else:
        all_partial_traces, gram_matrix_train = projected_kernel(train_features, ZZ_reps=ZZ_reps, ent_type=ent_type)

    t1 = time.time()
    print(t1-t0, 'sec to evaluate kernels')
    mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
    independent_entries = gram_matrix_train[mask]

    
    #SVC train and test
    svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

    svc.fit(gram_matrix_train, train_labels)

    if mps_sim:
        gram_matrix_test =  projected_kernel_test_mps(train_features, test_features, all_partial_traces, ZZ_reps=ZZ_reps, ent_type=ent_type)  
    else:
        gram_matrix_test =  projected_kernel_test(train_features, test_features, all_partial_traces, ZZ_reps=ZZ_reps, ent_type=ent_type) 

    score_kernel = svc.score(gram_matrix_test, test_labels)
    t2 = time.time()
    print('tot time: ', t2-t0, 'sec \n')
    print(f"Precomputed kernel classification test score: {score_kernel}\n")

    # Get predicted probabilities
    y_pred_probs = svc.predict(gram_matrix_test)

    # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Compute confusion matrix (normalized)
    labels = np.unique(test_labels)  # Dynamically determine class labels
    confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

    # Print Accuracy Per Class and Report
    print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
    print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))

    # if compute_entropy:
    #     return independent_entries, score_kernel, confusion, entanglement_entropy

    # else:
    #     return independent_entries, score_kernel, confusion


    return independent_entries, score_kernel, confusion, entanglement_entropy

    



############################ 
#SIMULATORS

def projected_kernel_mps(X, ZZ_reps=1, ent_type='linear'):

    print('MPS SIMULATOR')
    sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'

    N_FEATURES = X.shape[1]
    all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix

    for data_point in range(X.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        # fm_bound = fm.bind_parameters(X[data_point])
        fm_bound = fm.assign_parameters(X[data_point])
        qc.append(fm_bound, range(N_FEATURES))

        # Add one save_density_matrix per qubit, with a unique label
        for i in range(N_FEATURES):
            label = f"qubit_{i}"
            qc.save_density_matrix(qubits=[i], label=label)

        # Run once
        qc = transpile(qc, sim)
        result = sim.run(qc).result()
        data = result.data(0)

        # Extract each 1-qubit reduced density matrix
        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')
        for i in range(N_FEATURES):
            label = f"qubit_{i}"
            rho = data[label]
            partial_rhos[i]=rho

        all_partial_traces[data_point] = partial_rhos
            
    kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

    for row in range(kernel_matrix.shape[0]):
        for column in range(row+1,kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

    kernel_matrix += kernel_matrix.T
    kernel_matrix += np.identity(kernel_matrix.shape[0])

    return all_partial_traces, kernel_matrix



def projected_kernel_test_mps(X, Y, x_partial_traces, ZZ_reps=1, ent_type='linear'):

    print('MPS SIMULATOR')
    sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'

    N_FEATURES = X.shape[1]

    y_partial_traces = np.empty((Y.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix

    for data_point in range(Y.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        # fm_bound = fm.bind_parameters(X[data_point])
        fm_bound = fm.assign_parameters(Y[data_point])
        qc.append(fm_bound, range(N_FEATURES))

        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

        # Add one save_density_matrix per qubit, with a unique label
        for i in range(N_FEATURES):
            label = f"qubit_{i}"
            qc.save_density_matrix(qubits=[i], label=label)

        # Run once
        qc = transpile(qc, sim)
        result = sim.run(qc).result()
        data = result.data(0)

        # Extract each 1-qubit reduced density matrix
        partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')
        for i in range(N_FEATURES):
            label = f"qubit_{i}"
            rho = data[label]
            partial_rhos[i]=rho

        y_partial_traces[data_point] = partial_rhos

    all_partial_traces = np.concatenate((x_partial_traces, y_partial_traces), axis=0)

    kernel_matrix = np.zeros((Y.shape[0], X.shape[0]))

    for row in range(kernel_matrix.shape[0]):
        for column in range(kernel_matrix.shape[1]):
            kernel_matrix[row, column] = kernel_value(x1_id = X.shape[0] + row, x2_id = column, all_partial_traces_matrix = all_partial_traces)   
            
    return(kernel_matrix) 

def compute_entanglement_entropy(X, ZZ_reps=1, ent_type='linear'):

    #NEED TO ADD THE SAME FOR TEST FUNCTION!!!!!!!!!!!!!

    method = 'statevector' #'matrix_product_state' or 'statevector'
    device = 'GPU' #'CPU' or 'GPU'
    print('Simulator method device ', method, device)
    # sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'
    sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'

    print('sim', sim)
    N_FEATURES = X.shape[1]
    all_entropies = np.empty((X.shape[0]))

    for data_point in range(X.shape[0]):
        qc = QuantumCircuit(N_FEATURES)
        fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        # fm_bound = fm.bind_parameters(X[data_point])
        fm_bound = fm.assign_parameters(X[data_point])
        qc.append(fm_bound, range(N_FEATURES))

        # Save subsystem A (first half) density matrix
        half = N_FEATURES // 2
        qc.save_density_matrix(list(range(half)), label='subsystem_half')

        # Run once
        qc = transpile(qc, sim)
        result = sim.run(qc).result()
        data = result.data(0)

        reduced_rho = data['subsystem_half']

        all_entropies[data_point] = entropy(reduced_rho, base=2)
            
    entanglement_entropy = np.mean(all_entropies)

    return entanglement_entropy, all_entropies
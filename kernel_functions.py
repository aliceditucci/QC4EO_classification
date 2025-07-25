import numpy as np
# import cupy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from itertools import product
from joblib import Parallel, delayed
from itertools import combinations
from tqdm import tqdm
import time
import os

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from qiskit.quantum_info import DensityMatrix,partial_trace, entropy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator




# def fidelity_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type):
        
#     num_features = train_features.shape[1] #num qubits
#     m = train_features.shape[0]  #num data points
        
#     feature_map = ZZFeatureMap(feature_dimension=num_features, reps=ZZ_reps, entanglement=ent_type)
#     sampler = Sampler()
#     fidelity = ComputeUncompute(sampler=sampler)
#     kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

#     t0 = time.time()
#     gram_matrix_train = kernel.evaluate(x_vec = train_features)
#     t1 = time.time()
#     print(t1-t0, 'sec to evaluate kernels')
#     mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
#     independent_entries = gram_matrix_train[mask]

    
#     #SVC train and test
#     svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

#     svc.fit(gram_matrix_train, train_labels)

#     gram_matrix_test = kernel.evaluate(x_vec = test_features, y_vec=train_features)

#     score_kernel = svc.score(gram_matrix_test, test_labels)
#     t2 = time.time()
#     print('tot time: ', t2-t0, 'sec \n')
#     print(f"Precomputed kernel classification test score: {score_kernel}\n")

#     # Get predicted probabilities
#     y_pred_probs = svc.predict(gram_matrix_test)

#     # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
#     y_pred = (y_pred_probs > 0.5).astype(int).flatten()

#     # Compute confusion matrix (normalized)
#     labels = np.unique(test_labels)  # Dynamically determine class labels
#     confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

#     # Print Accuracy Per Class and Report
#     print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
#     print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))


#     return independent_entries, score_kernel, confusion


# ####################################################################

# #PROJECTED KERNELS

# def Schatten_2_norm(X):
#     X_squared = np.matmul(X.conj().T, X)
#     norm = np.sqrt(np.trace(X_squared))
#     if(np.imag(norm)>1e-10):
#         print("Non-zero imaginary part of norm")
#         return('Error')
#     return(np.real(norm))

# def kernel_value_from_exponent_norms(exponent_norms, gamma = 1):
#     x = np.power(exponent_norms,2).sum()
#     return(np.exp(-gamma*x))

# def kernel_value(x1_id, x2_id, all_partial_traces_matrix, gamma = 1):
#     exponent_norms = []
#     partial_rhos_1 = all_partial_traces_matrix[x1_id]
#     partial_rhos_2 = all_partial_traces_matrix[x2_id]

#     for i in range(all_partial_traces_matrix.shape[1]):
#         partial_rho_1 = partial_rhos_1[i]
#         partial_rho_2 = partial_rhos_2[i]
#         exponent_norms.append(Schatten_2_norm(partial_rho_1-partial_rho_2))
#     exponent_norms = np.array(exponent_norms)
    
#     return(kernel_value_from_exponent_norms(exponent_norms, gamma=gamma))

# def projected_kernel(X, ZZ_reps=1, ent_type='linear'):

#     N_FEATURES = X.shape[1]
#     all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
#     qubit_list = [i for i in range(N_FEATURES)]

#     for data_point in range(X.shape[0]):
#         qc = QuantumCircuit(N_FEATURES)
#         fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#         # fm_bound = fm.bind_parameters(X[data_point])
#         fm_bound = fm.assign_parameters(X[data_point])
#         qc.append(fm_bound, range(N_FEATURES))
#         rho = DensityMatrix.from_instruction(qc)

#         partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

#         for current_qubit in qubit_list:
#             list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
#             partial_rho=partial_trace(rho,list_to_trace_out).data
#             partial_rhos[current_qubit] = partial_rho

#         all_partial_traces[data_point] = partial_rhos

#     kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

#     for row in range(kernel_matrix.shape[0]):
#         for column in range(row+1,kernel_matrix.shape[1]):
#             kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

#     kernel_matrix += kernel_matrix.T
#     kernel_matrix += np.identity(kernel_matrix.shape[0])

#     return all_partial_traces, kernel_matrix

# def projected_kernel_test(X, Y, x_partial_traces, ZZ_reps=1, ent_type='linear'):

#     N_FEATURES = X.shape[1]

#     y_partial_traces = np.empty((Y.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
#     qubit_list = [i for i in range(N_FEATURES)]

#     for data_point in range(Y.shape[0]):
#         qc = QuantumCircuit(N_FEATURES)
#         fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#         # fm_bound = fm.bind_parameters(X[data_point])
#         fm_bound = fm.assign_parameters(Y[data_point])
#         qc.append(fm_bound, range(N_FEATURES))
#         rho = DensityMatrix.from_instruction(qc)

#         partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

#         for current_qubit in qubit_list:
#             list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
#             partial_rho=partial_trace(rho,list_to_trace_out).data
#             partial_rhos[current_qubit] = partial_rho

#         y_partial_traces[data_point] = partial_rhos

#     all_partial_traces = np.concatenate((x_partial_traces, y_partial_traces), axis=0)

#     kernel_matrix = np.zeros((Y.shape[0], X.shape[0]))

#     for row in range(kernel_matrix.shape[0]):
#         for column in range(kernel_matrix.shape[1]):
#             kernel_matrix[row, column] = kernel_value(x1_id = X.shape[0] + row, x2_id = column, all_partial_traces_matrix = all_partial_traces)   

#     # kernel_matrix += kernel_matrix.T
#     # kernel_matrix += np.identity(kernel_matrix.shape[0])
#     return(kernel_matrix) 

# def projected_kernel_with_entropy(X, ZZ_reps=1, ent_type='linear'):

#     N_FEATURES = X.shape[1]
#     all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix
#     qubit_list = [i for i in range(N_FEATURES)]

#     all_entropies = np.empty((X.shape[0]))

#     for data_point in range(X.shape[0]):
#         qc = QuantumCircuit(N_FEATURES)
#         fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#         # fm_bound = fm.bind_parameters(X[data_point])
#         fm_bound = fm.assign_parameters(X[data_point])
#         qc.append(fm_bound, range(N_FEATURES))
#         rho = DensityMatrix.from_instruction(qc)

#         partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

#         for current_qubit in qubit_list:
#             list_to_trace_out = [qubit_id for qubit_id in qubit_list if qubit_id != current_qubit]
#             partial_rho=partial_trace(rho,list_to_trace_out).data
#             partial_rhos[current_qubit] = partial_rho

#         all_partial_traces[data_point] = partial_rhos

#         reduced_rho = partial_trace(rho,range(round(N_FEATURES/2)))
#         all_entropies[data_point] = entropy(reduced_rho, base=2)

#     kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

#     for row in range(kernel_matrix.shape[0]):
#         for column in range(row+1,kernel_matrix.shape[1]):
#             kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

#     kernel_matrix += kernel_matrix.T
#     kernel_matrix += np.identity(kernel_matrix.shape[0])

#     entanglement_entropy = np.mean(all_entropies)

#     return all_partial_traces, kernel_matrix, entanglement_entropy

# ############################ 
# #SIMULATORS

# def projected_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type,  method, device):

#     print(method, device)
#     # num_features = train_features.shape[1] #num qubits
#     m = train_features.shape[0]  #num data points

#     t0 = time.time()
    
#     all_partial_traces, gram_matrix_train = projected_kernel_simulator(train_features, ZZ_reps=ZZ_reps, ent_type=ent_type, method=method, device=device)
    
#     t1 = time.time()
#     print(t1-t0, 'sec to evaluate kernels')
#     mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
#     independent_entries = gram_matrix_train[mask]

    
#     #SVC train and test
#     svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

#     svc.fit(gram_matrix_train, train_labels)

#     gram_matrix_test =  projected_kernel_test_simulator(train_features, test_features, all_partial_traces, ZZ_reps=ZZ_reps, ent_type=ent_type, method=method, device=device)  

#     score_kernel = svc.score(gram_matrix_test, test_labels)
#     t2 = time.time()
#     print('tot time: ', t2-t0, 'sec \n')
#     print(f"Precomputed kernel classification test score: {score_kernel}\n")

#     # Get predicted probabilities
#     y_pred_probs = svc.predict(gram_matrix_test)

#     # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
#     y_pred = (y_pred_probs > 0.5).astype(int).flatten()

#     # Compute confusion matrix (normalized)
#     labels = np.unique(test_labels)  # Dynamically determine class labels
#     confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

#     # Print Accuracy Per Class and Report
#     print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
#     print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))

#     # if compute_entropy:
#     #     return independent_entries, score_kernel, confusion, entanglement_entropy

#     # else:
#     #     return independent_entries, score_kernel, confusion


#     return independent_entries, score_kernel, confusion


# def projected_kernel_simulator(X, ZZ_reps=1, ent_type='linear', method = 'matrix_product_state', device = 'CPU'):

#     #NEED TO ADD THE SAME FOR TEST FUNCTION!!!!!!!!!!!!!

#     method = method  #'matrix_product_state' or 'statevector'
#     device = device #'CPU' or 'GPU'
#     # sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'
#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1]
#     all_partial_traces = np.empty((X.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix

#     for data_point in range(X.shape[0]):
#         qc = QuantumCircuit(N_FEATURES)

#         if ent_type is None:
#             fm = ZFeatureMap(feature_dimension=N_FEATURES, reps=1)

#         else:
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)

#         # fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#         # fm_bound = fm.bind_parameters(X[data_point])
#         fm_bound = fm.assign_parameters(X[data_point])
#         qc.append(fm_bound, range(N_FEATURES))

#         # Add one save_density_matrix per qubit, with a unique label
#         for i in range(N_FEATURES):
#             label = f"qubit_{i}"
#             qc.save_density_matrix(qubits=[i], label=label)

#         # Run once
#         qc = transpile(qc, sim)
#         result = sim.run(qc).result()
#         data = result.data(0)

#         # Extract each 1-qubit reduced density matrix
#         partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')
#         for i in range(N_FEATURES):
#             label = f"qubit_{i}"
#             rho = data[label]
#             partial_rhos[i]=rho

#         all_partial_traces[data_point] = partial_rhos
            
#     kernel_matrix = np.zeros((X.shape[0],X.shape[0]))

#     for row in range(kernel_matrix.shape[0]):
#         for column in range(row+1,kernel_matrix.shape[1]):
#             kernel_matrix[row, column] = kernel_value(x1_id = row, x2_id = column, all_partial_traces_matrix = all_partial_traces)

#     kernel_matrix += kernel_matrix.T
#     kernel_matrix += np.identity(kernel_matrix.shape[0])

#     return all_partial_traces, kernel_matrix


# def projected_kernel_test_simulator(X, Y, x_partial_traces, ZZ_reps=1, ent_type='linear', method = 'matrix_product_state', device = 'CPU'):

#     method = method  #'matrix_product_state' or 'statevector'
#     device = device #'CPU' or 'GPU'
#     # sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'
#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1]

#     y_partial_traces = np.empty((Y.shape[0],N_FEATURES,2,2), dtype = 'complex128') # four dimensions: 0-number of data points, 1-number of partial trace matrices, 2,3 - dimension of partial trace matrix

#     for data_point in range(Y.shape[0]):
#         qc = QuantumCircuit(N_FEATURES)

#         if ent_type == 'none':
#             fm = ZFeatureMap(feature_dimension=N_FEATURES, reps=1)
#         else:
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
        
#         # fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#         # fm_bound = fm.bind_parameters(X[data_point])
#         fm_bound = fm.assign_parameters(Y[data_point])
#         qc.append(fm_bound, range(N_FEATURES))

#         partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')

#         # Add one save_density_matrix per qubit, with a unique label
#         for i in range(N_FEATURES):
#             label = f"qubit_{i}"
#             qc.save_density_matrix(qubits=[i], label=label)

#         # Run once
#         qc = transpile(qc, sim)
#         result = sim.run(qc).result()
#         data = result.data(0)

#         # Extract each 1-qubit reduced density matrix
#         partial_rhos = np.empty((N_FEATURES, 2, 2), dtype = 'complex128')
#         for i in range(N_FEATURES):
#             label = f"qubit_{i}"
#             rho = data[label]
#             partial_rhos[i]=rho

#         y_partial_traces[data_point] = partial_rhos

#     all_partial_traces = np.concatenate((x_partial_traces, y_partial_traces), axis=0)

#     kernel_matrix = np.zeros((Y.shape[0], X.shape[0]))

#     for row in range(kernel_matrix.shape[0]):
#         for column in range(kernel_matrix.shape[1]):
#             kernel_matrix[row, column] = kernel_value(x1_id = X.shape[0] + row, x2_id = column, all_partial_traces_matrix = all_partial_traces)   
            
#     return(kernel_matrix) 

# def compute_entanglement_entropy(X, ZZ_reps=1, ent_type='linear'):

#     #NEED TO ADD THE SAME FOR TEST FUNCTION!!!!!!!!!!!!!

#     method = 'statevector' #'matrix_product_state' or 'statevector'
#     device = 'GPU' #'CPU' or 'GPU'
#     print('Simulator method device ', method, device)
#     # sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'
#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'

#     print('sim', sim)
#     N_FEATURES = X.shape[1]
#     all_entropies = np.zeros((X.shape[0]))

#     for data_point in range(X.shape[0]):
#         qc = QuantumCircuit(N_FEATURES)
#         fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#         # fm_bound = fm.bind_parameters(X[data_point])
#         fm_bound = fm.assign_parameters(X[data_point])
#         qc.append(fm_bound, range(N_FEATURES))

#         # Save subsystem A (first half) density matrix
#         half = N_FEATURES // 2
#         qc.save_density_matrix(list(range(half)), label='subsystem_half')

#         # Run once
#         qc = transpile(qc, sim)
#         result = sim.run(qc).result()
#         data = result.data(0)

#         reduced_rho = data['subsystem_half']

#         all_entropies[data_point] = entropy(reduced_rho, base=2)
            
#     entanglement_entropy = np.mean(all_entropies)

#     return entanglement_entropy, all_entropies

# def fidelity_kernels_simulator(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type , method = 'matrix_product_state', device = 'CPU'):

#     print(method, device)
#     # num_features = train_features.shape[1] #num qubits
#     m = train_features.shape[0]  #num data points

#     t0 = time.time()
    
#     gram_matrix_train = get_fidelity_kernel_entries(train_features, ZZ_reps=ZZ_reps, ent_type=ent_type, method=method, device=device)

#     t1 = time.time()
#     print(t1-t0, 'sec to evaluate kernels')
#     mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
#     independent_entries = gram_matrix_train[mask]

#     #SVC train and test
#     svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

#     svc.fit(gram_matrix_train, train_labels)

#     gram_matrix_test =  get_fidelity_kernel_entries_test(train_features, test_features, ZZ_reps=ZZ_reps, ent_type=ent_type, method=method, device=device) 

#     score_kernel = svc.score(gram_matrix_test, test_labels)
#     t2 = time.time()
#     print('tot time: ', t2-t0, 'sec \n')
#     print(f"Precomputed kernel classification test score: {score_kernel}\n")

#     # Get predicted probabilities
#     y_pred_probs = svc.predict(gram_matrix_test)

#     # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
#     y_pred = (y_pred_probs > 0.5).astype(int).flatten()

#     # Compute confusion matrix (normalized)
#     labels = np.unique(test_labels)  # Dynamically determine class labels
#     confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

#     # Print Accuracy Per Class and Report
#     print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
#     print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))

#     return independent_entries, score_kernel, confusion


# def get_fidelity_kernel_entries(X, ZZ_reps, ent_type, method = 'statevector', device='CPU'):

#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1] #num qubits
#     m = X.shape[0]  #num data points
    
#     kernel_matrix = np.zeros((m,m))

#     for data_point in range(m):

#         for data_point2 in range(data_point+1, m):
#             qc1 = QuantumCircuit(N_FEATURES)
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#             # fm_bound = fm.bind_parameters(X[data_oint])
#             fm_bound = fm.assign_parameters(X[data_point])
#             qc1.append(fm_bound, range(N_FEATURES))

#             qc2 = QuantumCircuit(N_FEATURES)
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#             # fm_bound = fm.bind_parameters(X[data_point])
#             fm_bound = fm.assign_parameters(X[data_point2])
#             qc2.append(fm_bound, range(N_FEATURES))

#             qc = qc1.compose(qc2.inverse())
#             qc.save_statevector(label=f'statevector')

#             ####################
#             #IF LATER WE WANT TO DO SHOTS WE CAN SUE THIS FUNCTION

#             # options = {
#             # "backend_options": {
#             #     "method": method,
#             #     "device": device,
#             #     }}
#             # sampler = SamplerV2(options=options)
#             # print("Used backend", sampler._backend.configuration().to_dict()['backend_name'])
#             # print("Default shots:", sampler.default_shots)

#             # fidelity = ComputeUncompute(sampler=sampler)
#             # kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

#             # qc.measure_all()

#             # qc = transpile(qc, backend=sampler._backend, optimization_level=3)
#             # result  = sampler.run([qc], shots = 100000).result()
#             # # result  = sampler.run([qc]).result()

#             # # print(result.quasi_dists)
#             # # pub_result = result[0]
#             # # pub_result.data.meas.get_counts()
#             # # counts = pub_result.data.meas.get_counts()
#             # # print(pub_result.data.meas)
#             # # print(counts)
#             # quasis = post_process(result)
#             # print(quasis)
#             # fidelity = quasis.get(0, 0)
#             # kernel_matrix[data_point, data_point2] = fidelity
            
#             qc = transpile(qc, sim, optimization_level=3)
#             result = sim.run(qc).result()
#             state = result.data(0)['statevector']
#             fidelity = state.probabilities()[0]
#             kernel_matrix[data_point, data_point2] = fidelity


#     kernel_matrix += kernel_matrix.T
#     kernel_matrix += np.identity(kernel_matrix.shape[0])

#     return kernel_matrix


# def get_fidelity_kernel_entries_test(X, Y, ZZ_reps, ent_type, method = 'statevector', device='CPU'):
#     # X = train features
#     # Y = test features

#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1] #num qubits
#     m = X.shape[0]  #num train data points
    
#     kernel_matrix = np.zeros((Y.shape[0], m))

#     for row in range(kernel_matrix.shape[0]):
#         for column in range(kernel_matrix.shape[1]):

#             qc1 = QuantumCircuit(N_FEATURES)
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#             # fm_bound = fm.bind_parameters(X[data_oint])
#             fm_bound = fm.assign_parameters(X[row])
#             qc1.append(fm_bound, range(N_FEATURES))

#             qc2 = QuantumCircuit(N_FEATURES)
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#             # fm_bound = fm.bind_parameters(X[data_point])
#             fm_bound = fm.assign_parameters(X[column])
#             qc2.append(fm_bound, range(N_FEATURES))

#             qc = qc1.compose(qc2.inverse())
#             qc.save_statevector(label=f'statevector')
            
#             qc = transpile(qc, sim, optimization_level=3)
#             result = sim.run(qc).result()
#             state = result.data(0)['statevector']
#             fidelity = state.probabilities()[0]
#             kernel_matrix[row, column] = fidelity

#     return kernel_matrix


# def fk_compute_entanglement_entropy(X, ZZ_reps=1, ent_type='linear'):

#     #NEED TO ADD THE SAME FOR TEST FUNCTION!!!!!!!!!!!!!

#     method = 'statevector' #'matrix_product_state' or 'statevector'
#     # device = 'GPU' #'CPU' or 'GPU'
#     device = 'GPU'
#     print(method, device)
#     # sim = AerSimulator(method='matrix_product_state') #'matrix_product_state' or 'statevector'
#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1]
#     m = X.shape[0]
#     points = int(m*(m-1)/2)
#     all_entropies = np.zeros((points))
#     # all_entropies = np.zeros((m-1))

#     # for data_point in range(1):
 
#     for data_point in range(m):

#         for data_point2 in range(data_point+1, m):
#             qc1 = QuantumCircuit(N_FEATURES)
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#             # fm_bound = fm.bind_parameters(X[data_oint])
#             fm_bound = fm.assign_parameters(X[data_point])
#             qc1.append(fm_bound, range(N_FEATURES))

#             qc2 = QuantumCircuit(N_FEATURES)
#             fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)
#             # fm_bound = fm.bind_parameters(X[data_point])
#             fm_bound = fm.assign_parameters(X[data_point2])
#             qc2.append(fm_bound, range(N_FEATURES))

#             qc = qc1.compose(qc2.inverse())

#             # Save subsystem A (first half) density matrix
#             half = N_FEATURES // 2
#             qc.save_density_matrix(list(range(half)), label='subsystem_half')

#             qc = transpile(qc, sim, optimization_level=3)
#             result = sim.run(qc).result()
#             data = result.data(0)

#             reduced_rho = data['subsystem_half']

#             all_entropies[data_point] = entropy(reduced_rho, base=2)
            
#     entanglement_entropy = np.mean(all_entropies)
#     return entanglement_entropy, all_entropies




# ##########################################################

# def fidelity_kernels_parallel(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type , method = 'matrix_product_state', device = 'CPU'):

#     print(method, device)
#     # num_features = train_features.shape[1] #num qubits
#     m = train_features.shape[0]  #num data points

#     t0 = time.time()
    
#     gram_matrix_train = get_entries(train_features, ZZ_reps=ZZ_reps, ent_type=ent_type, method=method, device=device)

#     t1 = time.time()
#     print(t1-t0, 'sec to evaluate kernels')
#     mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
#     independent_entries = gram_matrix_train[mask]

#     #SVC train and test
#     svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

#     svc.fit(gram_matrix_train, train_labels)

#     gram_matrix_test =  get_entries_test(train_features, test_features, ZZ_reps=ZZ_reps, ent_type=ent_type, method=method, device=device) 

#     score_kernel = svc.score(gram_matrix_test, test_labels)
#     t2 = time.time()
#     print('tot time: ', t2-t0, 'sec \n')
#     print(f"Precomputed kernel classification test score: {score_kernel}\n")

#     # Get predicted probabilities
#     y_pred_probs = svc.predict(gram_matrix_test)

#     # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
#     y_pred = (y_pred_probs > 0.5).astype(int).flatten()

#     # Compute confusion matrix (normalized)
#     labels = np.unique(test_labels)  # Dynamically determine class labels
#     confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

#     # Print Accuracy Per Class and Report
#     print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
#     print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))

#     return independent_entries, score_kernel, confusion


# def compute_entry(i, j, x, y, N_FEATURES, ZZ_reps, ent_type, method, device):
#     sim = AerSimulator(method=method, device=device)

#     fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)

#     qc1 = QuantumCircuit(N_FEATURES)
#     qc1.append(fm.assign_parameters(x), range(N_FEATURES))

#     qc2 = QuantumCircuit(N_FEATURES)
#     qc2.append(fm.assign_parameters(y), range(N_FEATURES))

#     qc = qc1.compose(qc2.inverse())
#     qc.save_statevector()
#     # qc = transpile(qc, sim, optimization_level=3)
#     qc = transpile(qc, sim)

#     result = sim.run(qc).result()
#     state = result.data(0)['statevector']
#     fidelity = state.probabilities()[0]

#     return i, j, fidelity


# def get_entries(X, ZZ_reps, ent_type, method = 'statevector', device='CPU'):

#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1]
#     m = X.shape[0]  #num data points
    
#     kernel_matrix = np.ones((m,m)) 

#     indices = list(combinations(range(m), 2))

#     entries = Parallel(n_jobs=-1)(
#         delayed(compute_entry)(i, j , X[i], X[j], N_FEATURES, ZZ_reps, ent_type, method, device)
#         for i, j in indices
#         # for i, j in tqdm(indices, desc="Computing kernel entries")
#     )

#     for i, j, val in entries:
#         kernel_matrix[i, j] = val
#         if i != j:
#             kernel_matrix[j, i] = val

#     return kernel_matrix


# def get_entries_test(X, Y, ZZ_reps, ent_type, method = 'statevector', device='CPU'):

#     sim = AerSimulator(method=method, device=device) #'matrix_product_state' or 'statevector'
#     print("Used backend", sim.configuration().to_dict()['backend_name'])

#     N_FEATURES = X.shape[1]
#     m = X.shape[0]  #num data points
#     n = Y.shape[0]
    
#     kernel_matrix = np.ones((n,m)) 

#     indices = list(product(range(n), range(m)))

#     entries = Parallel(n_jobs=-1)(
#         delayed(compute_entry)(i, j, Y[i], X[j], N_FEATURES, ZZ_reps, ent_type, method, device)
#         for i, j in indices
#         # for i, j in tqdm(indices, desc="Computing kernel entries")
#     )

#     for i, j, val in entries:
#         kernel_matrix[i, j] = val

#     return kernel_matrix



# ##########################################
# #BATCH STATE VECTOR

# def fidelity_kernels_batch(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type , method = 'matrix_product_state', device = 'CPU'):

#     print(method, device)
#     # num_features = train_features.shape[1] #num qubits
#     m = train_features.shape[0]  #num data points

#     t0 = time.time()
    
#     gram_matrix_train = get_kernel_entries_frombatch(train_features, ZZ_reps, ent_type, method, device)

#     t1 = time.time()
#     print(t1-t0, 'sec to evaluate kernels')
#     mask = np.triu_indices(m,k=1) #returns the indices of the upper-triangular part of an (m x m) square matrix.
#     independent_entries = gram_matrix_train[mask]

#     #SVC train and test
#     svc = SVC(kernel="precomputed", random_state=42) #Remove , random_state=42 for not deterministic

#     svc.fit(gram_matrix_train, train_labels)

#     gram_matrix_test =  get_kernel_entries_frombatch_test(train_features, test_features, ZZ_reps, ent_type, method, device)

#     score_kernel_train = svc.score(gram_matrix_train, train_labels) #train accuracy
#     score_kernel = svc.score(gram_matrix_test, test_labels) #validation accuracy

#     t2 = time.time()
#     print('tot time: ', t2-t0, 'sec \n')
#     print(f"Precomputed kernel classification test score: {score_kernel}\n")

#     # Get predicted probabilities
#     y_pred_probs = svc.predict(gram_matrix_test)

#     # Convert probabilities to class labels (assuming binary classification with threshold 0.5)
#     y_pred = (y_pred_probs > 0.5).astype(int).flatten()

#     # Compute confusion matrix (normalized)
#     labels = np.unique(test_labels)  # Dynamically determine class labels
#     confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

#     # Print Accuracy Per Class and Report
#     print('Accuracy per class:', confusion.diagonal(), 'Mean Accuracy:', confusion.diagonal().mean())
#     print(classification_report(test_labels, y_pred, labels=labels, target_names=[str(label) for label in labels], digits=4))

#     return independent_entries, gram_matrix_test, score_kernel, score_kernel_train, confusion

# def batch_generator(data, batch_size):
#     """Yield batches of data."""
#     for i in range(0, len(data), batch_size):
#         yield data[i:i+batch_size]


# def get_state(x, N_FEATURES, ZZ_reps, ent_type, method, device):
#     sim = AerSimulator(method=method, device=device)

#     if ent_type == 'none':
#         fm = ZFeatureMap(feature_dimension=N_FEATURES, reps=1)

#     else:
#         fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=ZZ_reps, entanglement=ent_type)


#     qc1 = QuantumCircuit(N_FEATURES)
#     qc1.append(fm.assign_parameters(x), range(N_FEATURES))
#     qc1.save_statevector()

#     qc1 = transpile(qc1, sim, optimization_level=3)

#     result = sim.run(qc1).result()
#     state = result.data(0)['statevector']

#     return state 


# def compute_all_states_batched(X, ZZ_reps, ent_type, method='statevector', device='CPU', n_jobs=4, batch_size=100):

#     N_FEATURES = X.shape[1]
#     all_states = []

#     for batch in batch_generator(X, batch_size):
#         # Compute states for this batch in parallel
#         batch_states = Parallel(n_jobs=n_jobs)(
#             delayed(get_state)(x, N_FEATURES, ZZ_reps, ent_type, method, device) for x in batch
#         )
#         all_states.extend(batch_states)

#     return np.array(all_states)

# def get_kernel_entries_frombatch(X, ZZ_reps, ent_type, method='statevector', device='CPU', n_jobs=4, batch_size=100):

#     X_states = compute_all_states_batched(X, ZZ_reps, ent_type, method, device, n_jobs, batch_size)
#     kernel_matrix = np.abs(X_states @ X_states.conj().T) ** 2

#     return kernel_matrix

# def get_kernel_entries_frombatch_test(X, Y, ZZ_reps, ent_type, method='statevector', device='CPU', n_jobs=4, batch_size=100):

#     X_states = compute_all_states_batched(X, ZZ_reps, ent_type, method, device, n_jobs, batch_size)
#     Y_states = compute_all_states_batched(Y, ZZ_reps, ent_type, method, device, n_jobs, batch_size)
#     kernel_matrix = np.abs(Y_states @ X_states.conj().T) ** 2

#     return kernel_matrix

# #####################################
# #Downscale dataset

# def reconstruct_gram_matrix(independent_entries):
    
#     m = independent_entries.shape[0]
#     print('num points', m)
#     gram_matrix = np.zeros((m, m))
    
#     # Fill upper triangle (excluding diagonal)
#     mask = np.triu_indices(m, k=1)
#     gram_matrix[mask] = independent_entries
    
#     # Reflect upper triangle to lower triangle (symmetry)
#     gram_matrix += gram_matrix.T
#     np.fill_diagonal(gram_matrix, 1.0) 

#     return gram_matrix



#############################################


class EvaluateKernels:
    def __init__(self, method='statevector', device='CPU', ZZ_reps=1, ent_type='linear',
                 n_qubits=None, ansatz_type=None, n_jobs=4, batch_size=100, kernel_type='fidelity', gamma=1.0):
        
        """
        Args:
            method: str, kernel evaluation method ('matrix_product_state' or 'statevector')
            device: str, execution device ('CPU' or 'GPU')
            ZZ_reps: ansatz depth or reps for ZZFeatureMap or custom circuit
            ent_type: entanglement type (e.g., 'linear', 'full', etc.)
            n_qubits: int, number of qubits
            n_jobs: int, num of jobs run in parallel in fidelity kernel cpu
            batch_size: int, batch of fidelity kernel computation
            kernel_type: str, type of kernel 'fidelity' or 'linear'
            gamma: float, decay of the Gaussian-like projected kernel
        """
                
        self.method = method
        self.device = device
        self.ZZ_reps = ZZ_reps
        self.ent_type = ent_type
        self.n_qubits = n_qubits
        # self.ansatz_type = ansatz_type
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.kernel_type = kernel_type
        self.gamma = gamma
    
    def compute_kernel_matrices(self, X, Y):

        if self.kernel_type == 'fidelity':
            independent_entries, gram_matrix_test = self.fidelity_kernels_batch(X, Y)

        elif self.kernel_type == 'projected':
            independent_entries, gram_matrix_test = self.projected_kernels(X, Y)

        else:
            raise ValueError(f"Unknown kernel_type '{self.kernel_type}'. Use 'fidelity' or 'projected'.")

        return independent_entries, gram_matrix_test

    def fidelity_kernels_batch(self, train_features, test_features):
        print(self.method, self.device)
        m = train_features.shape[0]

        t0 = time.time()
        gram_matrix_train = self.get_kernel_entries_frombatch(train_features)
        t1 = time.time()
        print(f"{t1 - t0:.2f} sec to evaluate training kernels")

        mask = np.triu_indices(m, k=1)
        independent_entries = gram_matrix_train[mask]

        gram_matrix_test = self.get_kernel_entries_frombatch_test(train_features, test_features)

        t2 = time.time()
        print(f"Total kernel computation time: {t2 - t0:.2f} sec\n")

        return independent_entries, gram_matrix_test

    def batch_generator(self, data):
        for i in range(0, len(data), self.batch_size):
            yield data[i:i + self.batch_size]

    def get_state(self, x, n_features):
        sim = AerSimulator(method=self.method, device=self.device)

        if self.ent_type == 'none':
            fm = ZFeatureMap(feature_dimension=n_features, reps=1)
        else:
            fm = ZZFeatureMap(feature_dimension=n_features, reps=self.ZZ_reps, entanglement=self.ent_type)

        qc = QuantumCircuit(n_features)
        qc.append(fm.assign_parameters(x), range(n_features))
        qc.save_statevector()
        qc = transpile(qc, sim, optimization_level=3)
        result = sim.run(qc).result()

        return result.data(0)['statevector']

    def compute_all_states_batched(self, X):
        n_features = X.shape[1]
        all_states = []

        for batch in self.batch_generator(X):
            batch_states = Parallel(n_jobs=self.n_jobs)(
                delayed(self.get_state)(x, n_features) for x in batch
            )
            all_states.extend(batch_states)

        return np.array(all_states)

    def get_kernel_entries_frombatch(self, X):
        X_states = self.compute_all_states_batched(X)
        kernel_matrix = np.abs(X_states @ X_states.conj().T) ** 2
        return kernel_matrix

    def get_kernel_entries_frombatch_test(self, X, Y):
        X_states = self.compute_all_states_batched(X)
        Y_states = self.compute_all_states_batched(Y)
        kernel_matrix = np.abs(Y_states @ X_states.conj().T) ** 2
        return kernel_matrix
    
    def projected_kernels(self, train_features, test_features):
        m = train_features.shape[0]
        t0 = time.time()

        x_partial_traces, gram_matrix_train = self.projected_kernel_simulator(train_features)
        t1 = time.time()
        print(f"{t1 - t0:.2f} sec to evaluate training projected kernels")

        mask = np.triu_indices(m, k=1)
        independent_entries = gram_matrix_train[mask]

        gram_matrix_test = self.projected_kernel_test_simulator(train_features, test_features, x_partial_traces)

        t2 = time.time()
        print(f"Total projected kernel computation time: {t2 - t0:.2f} sec\n")

        return independent_entries, gram_matrix_test

    def projected_kernel_simulator(self, X):
        sim = AerSimulator(method=self.method, device=self.device)
        N = X.shape[1]
        n_data = X.shape[0]

        all_partial_traces = np.empty((n_data, N, 2, 2), dtype='complex128')

        for i in range(n_data):
            qc = QuantumCircuit(N)

            if self.ent_type == 'none':
                fm = ZFeatureMap(feature_dimension=N, reps=1)
            else:
                fm = ZZFeatureMap(feature_dimension=N, reps=self.ZZ_reps, entanglement=self.ent_type)

            qc.append(fm.assign_parameters(X[i]), range(N))
            for j in range(N):
                qc.save_density_matrix(qubits=[j], label=f"q{j}")

            qc = transpile(qc, sim)
            result = sim.run(qc).result()
            data = result.data(0)

            for j in range(N):
                all_partial_traces[i, j] = data[f"q{j}"]

        K = np.zeros((n_data, n_data))
        for i in range(n_data):
            for j in range(i + 1, n_data):
                K[i, j] = self._kernel_value(i, j, all_partial_traces)
        K += K.T
        np.fill_diagonal(K, 1.0)

        return all_partial_traces, K

    def projected_kernel_test_simulator(self, X, Y, x_partial_traces):
        sim = AerSimulator(method=self.method, device=self.device)
        N = X.shape[1]
        n_test = Y.shape[0]
        n_train = X.shape[0]

        y_partial_traces = np.empty((n_test, N, 2, 2), dtype='complex128')

        for i in range(n_test):
            qc = QuantumCircuit(N)

            if self.ent_type == 'none':
                fm = ZFeatureMap(feature_dimension=N, reps=1)
            else:
                fm = ZZFeatureMap(feature_dimension=N, reps=self.ZZ_reps, entanglement=self.ent_type)

            qc.append(fm.assign_parameters(Y[i]), range(N))
            for j in range(N):
                qc.save_density_matrix(qubits=[j], label=f"q{j}")

            qc = transpile(qc, sim)
            result = sim.run(qc).result()
            data = result.data(0)

            for j in range(N):
                y_partial_traces[i, j] = data[f"q{j}"]

        all_traces = np.concatenate([x_partial_traces, y_partial_traces], axis=0)
        K = np.zeros((n_test, n_train))

        for i in range(n_test):
            for j in range(n_train):
                K[i, j] = self._kernel_value(n_train + i, j, all_traces)

        return K

    def _schatten_2_norm(self, X):
        X_squared = np.matmul(X.conj().T, X)
        norm = np.sqrt(np.trace(X_squared))
        if np.imag(norm) > 1e-10:
            print("Non-zero imaginary part of norm")
            raise ValueError("Imaginary part of Schatten-2 norm is too large.")
        return np.real(norm)

    def _kernel_value(self, x1_id, x2_id, all_partial_traces_matrix):
        exponent_norms = []
        rhos1 = all_partial_traces_matrix[x1_id]
        rhos2 = all_partial_traces_matrix[x2_id]
        for i in range(rhos1.shape[0]):
            norm = self._schatten_2_norm(rhos1[i] - rhos2[i])
            exponent_norms.append(norm)
        exponent_norms = np.array(exponent_norms)
        return np.exp(-self.gamma * np.sum(exponent_norms ** 2))
    
    def save_kernel_data(self, filename, independent_entries, gram_matrix_test, train_labels, test_labels):
        """
        Save kernel matrices and labels to a .npz file for later SVM training/testing.
        
        Args:
            filename (str): Path to the output file (should end in .npz)
            independent_entries (np.ndarray): Upper-triangular entries of train kernel
            gram_matrix_test (np.ndarray): Test kernel matrix
            train_labels (np.ndarray): Training labels
            test_labels (np.ndarray): Test labels
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(
            filename,
            independent_entries=independent_entries,
            gram_matrix_test=gram_matrix_test,
            train_labels=train_labels,
            test_labels=test_labels
        )
        print(f"Saved kernel data to {filename}")


#############################################

def reconstruct_gram_matrix(independent_entries):
    
    num_entries = independent_entries.shape[0]
    m = int((1 + np.sqrt(1 + 8 * num_entries)) / 2)

    print('num points', m)
    gram_matrix = np.zeros((m, m))
    
    # Fill upper triangle (excluding diagonal)
    mask = np.triu_indices(m, k=1)
    gram_matrix[mask] = independent_entries
    
    # Reflect upper triangle to lower triangle (symmetry)
    gram_matrix += gram_matrix.T
    np.fill_diagonal(gram_matrix, 1.0) 

    return gram_matrix

def evaluate_svm(independent_entries, train_labels, gram_matrix_test, test_labels):

    gram_matrix_train = reconstruct_gram_matrix(independent_entries)
    svc = SVC(kernel="precomputed", random_state=42)
    svc.fit(gram_matrix_train, train_labels)

    score_kernel_train = svc.score(gram_matrix_train, train_labels)
    score_kernel = svc.score(gram_matrix_test, test_labels)

    print(f"Train accuracy: {score_kernel_train:.4f}")
    print(f"Test accuracy: {score_kernel:.4f}\n")

    y_pred_probs = svc.predict(gram_matrix_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    labels = np.unique(test_labels)
    confusion = confusion_matrix(test_labels, y_pred, labels=labels, normalize='true')

    print("Accuracy per class:", confusion.diagonal())
    print("Mean Accuracy:", confusion.diagonal().mean())
    print(classification_report(test_labels, y_pred, labels=labels,
                                target_names=[str(label) for label in labels], digits=4))

    return score_kernel, score_kernel_train, confusion

  
def save_svm_data(filename, score, score_train, confusion):
    """
    Save kernel matrices and labels to a .npz file for later SVM training/testing.
    
    Args:
        filename (str): Path to the output file (should end in .npz)
        independent_entries (np.ndarray): Upper-triangular entries of train kernel
        gram_matrix_test (np.ndarray): Test kernel matrix
        train_labels (np.ndarray): Training labels
        test_labels (np.ndarray): Test labels
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(
        filename,
        score = score,
        score_train = score_train, 
        confusion = confusion
    )
    print(f"Saved svm data to {filename}")

# ##### USE#######
# ek = EvaluateKernels(method='statevector', ZZ_reps=2, batch_size=64, n_jobs=8)
# ek.fidelity_kernels_batch(train_X, train_y, test_X, test_y)
# # Compute kernel matrices
# ind_entries, gram_train, gram_test = ek.compute_kernel_matrices(train_X, test_X)

# # Evaluate using SVM
# score_test, score_train, conf = ek.evaluate_svm(gram_train, train_y, gram_test, test_y)
# data = np.load("output/kernels_data.npz")
# ind_entries = data['independent_entries']
# gram_matrix_test = data['gram_matrix_test']
# train_labels = data['train_labels']
# test_labels = data['test_labels']


# scalers = {
#     'standard': StandardScaler(),
#     'minmax': MinMaxScaler(),
#     'robust': RobustScaler()
# }

# for name, scaler in scalers.items():
#     print(f"\n--- Running with {name} scaler ---")
    
#     X_train_scaled = scaler.fit_transform(train_X)
#     X_test_scaled = scaler.transform(test_X)
    
#     ind_entries, gram_train, gram_test = ek.compute_kernel_matrices(X_train_scaled, X_test_scaled)
    
#     ek.save_kernel_data(f"results/kernels_{name}.npz", ind_entries, gram_test, train_y, test_y)

#     # Or: run evaluation inline
#     # score_test, score_train, confusion = ek.evaluate_svm(gram_train, train_y, gram_test, test_y)
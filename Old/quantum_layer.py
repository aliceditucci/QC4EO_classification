import numpy as np 
# import pennylane.numpy as np  # PennyLane-compatible NumPy
import itertools

import torch
from torch.autograd import Function

import qiskit
from qiskit.circuit import Parameter
from qiskit import transpile #qiskit>=1 version

import pennylane as qml

#Circuit type I
class QiskitCircuit():
    
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.thetas = {k : Parameter('Theta'+str(k))for k in range(1*self.n_qubits)}
        
        all_qubits = [i for i in range(n_qubits)]
        self.circuit.h(all_qubits)
        self.circuit.barrier()

        #self.circuit.h(0)

        for k in range(0, 4):
          self.circuit.ry(self.thetas[k], k)
        

        self.circuit.measure_all()
        # ---------------------------
        
        self.backend = backend
        self.shots = shots

        self.QC_OUTPUTS = self.create_QC_OUTPUTS(self.n_qubits)

    # create list of all possible outputs of quantum circuit (2**NUM_QUBITS possible)
    def create_QC_OUTPUTS(self, n_qubits):
        measurements = list(itertools.product([0, 1], repeat=n_qubits))
        return [''.join([str(bit) for bit in measurement]) for measurement in measurements] 
        
    def N_qubit_expectation_Z(self,counts, shots, n_qubits, QC_OUTPUTS):
        expects = np.zeros(len(QC_OUTPUTS))
        for k in range(len(QC_OUTPUTS)):
            key = QC_OUTPUTS[k]
            perc = counts.get(key, 0)/shots
            expects[k] = perc
        return expects
    
    def run(self, i):
        params = i
        #print('params = {}'.format(len(params)))
        # backend = Aer.get_backend('qasm_simulator')
    
        # job_sim = execute(self.circuit,
        #                       self.backend,
        #                       shots=self.shots,
        #                       parameter_binds = [{self.thetas[k] : params[k].item() for k in range(1*NUM_QUBITS)}])
        bound_circuit = self.circuit.assign_parameters({self.thetas[k] : params[k].item() for k in range(1*self.n_qubits)}) 
        new_circuit = transpile(bound_circuit, self.backend)
        job_sim = self.backend.run(new_circuit, shots=self.shots)
        result_sim = job_sim.result()
        # counts = result_sim.get_counts(self.circuit) #changed because it was giving an error
        counts = result_sim.get_counts()
        
        return self.N_qubit_expectation_Z(counts, self.shots, self.n_qubits, self.QC_OUTPUTS)
    
#TorchCircuit type I 
class TorchCircuit(Function):    

    @staticmethod
    # def forward(ctx, i, qiskit_circ):
    def forward(ctx, i, NUM_QUBITS, SIMULATOR, NUM_SHOTS, shift):
        if not hasattr(ctx, 'QiskitCirc'):
            # ctx.QiskitCirc = qiskit_circ
            ctx.QiskitCirc = QiskitCircuit(NUM_QUBITS, SIMULATOR, shots=NUM_SHOTS)
            ctx.shift = shift
        exp_value = ctx.QiskitCirc.run(i)
        
        result = torch.tensor([exp_value])
  
        ctx.save_for_backward(result, i)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        
        forward_tensor, i = ctx.saved_tensors
#         print('forward_tensor = {}'.format(forward_tensor))
        input_numbers = i
#         print('input_numbers = {}'.format(input_numbers))
        gradients = torch.Tensor()
        
        # print(f"Shape of input tensor i: {i.shape}")
        # print(f"Values of input tensor i: {i}")
        # print('ctx.QiskitCirc.n_qubits', ctx.QiskitCirc.n_qubits)

        for k in range(1*ctx.QiskitCirc.n_qubits):
            shift_right = input_numbers.detach().clone()
            # shift_right[k] = shift_right[k] + ctx.QiskitCirc.shift
            shift_right[k] = shift_right[k] + ctx.shift
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - ctx.shift
            
#             print('shift_right = {}, shift_left = {}'.format(shift_right, shift_left))
            
            expectation_right = ctx.QiskitCirc.run(shift_right)
            expectation_left  = ctx.QiskitCirc.run(shift_left)
#             print('expectation_right = {}, \nexpectation_left = {}'.format(expectation_right, expectation_left))
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            # rescale gradient
#             gradient = gradient / torch.norm(gradient)
#             print('gradient for k={}: {}'.format(k, gradient))
            gradients = torch.cat((gradients, gradient.float()))
            
        result = torch.Tensor(gradients)
        # print('gradients = {}'.format(result))
        # print('grad_output = {}'.format(grad_output))
        # print('output', ((result.float() * grad_output.float()).T).shape)
        return (result.float() * grad_output.float()).T, None, None, None, None
    




# dev = qml.device('lightning.qubit', wires=NUM_QUBITS)
dev = qml.device('default.qubit', wires=4)

def embedding_VQC(data):
    for idx, x in enumerate(data):
      qml.RX(x,wires=idx)
    for idx in range(len(data)-1):
      qml.CNOT(wires=[idx,idx+1])

#SO(4) circuit
def conv(qubit0, qubit1, params):
    qml.RZ(np.pi/4, wires=qubit0)
    qml.RZ(np.pi/4, wires=qubit1)
    qml.RY(np.pi/2, wires=qubit1)
    qml.CNOT(wires=[qubit1,qubit0])
    qml.Rot(params[0], params[1], params[2], wires=qubit0)
    qml.Rot(params[3], params[4], params[5], wires=qubit1)
    qml.CNOT(wires=[qubit1,qubit0])
    qml.RY(-np.pi/2, wires=qubit1)
    qml.RZ(-np.pi/4, wires=qubit1)
    qml.RZ(-np.pi/4, wires=qubit0)

def pool(sinkbit, targetbit, params):
    qml.Rot(params[0], params[1], params[2], wires=sinkbit)
    qml.Rot(params[3], params[4], params[5], wires=targetbit)
    qml.CNOT(wires=[sinkbit,targetbit])
    qml.Rot(params[6], params[7], params[8], wires=targetbit)

# General function for applying convolution and pooling
def apply_layer(params, wires):
    # Apply convolution on even-indexed qubits
    for idx in range(0, len(wires) - 1, 2):
        conv(wires[idx], wires[idx + 1], params[:6])

    # Apply convolution on odd-indexed qubits
    for idx in range(1, len(wires) - 1, 2):
        conv(wires[idx], wires[idx + 1], params[:6])

    # Apply convolution between last and first qubit
    conv(wires[-1], wires[0], params[:6])

    # Apply pooling
    half_size = len(wires) // 2
    for idx in range(half_size):
        pool(wires[idx], wires[idx + half_size], params[6:15]) 

#THIS IS AFTER DEA
# #pooling circuit obtained from https://github.com/ML4SCI/QML-hands-on/blob/main/notebooks/4_QCNN_MNIST.ipynb      
# def pool_lessZ(sinkbit, targetbit, params):
#     qml.Rot(params[0], params[1], params[2], wires=sinkbit)
#     qml.Rot(params[3], params[4], params[5], wires=targetbit)
#     qml.CNOT(wires=[sinkbit,targetbit])
#     #qml.Rot(params[6], params[7], params[8], wires=targetbit)
#     qml.RZ(params[6], wires=targetbit)
#     qml.RY(params[7], wires=targetbit)
    
# #pooling circuit obtained from https://github.com/ML4SCI/QML-hands-on/blob/main/notebooks/4_QCNN_MNIST.ipynb      
# def pool_lessRot(sinkbit, targetbit, params):
#     qml.Rot(params[0], params[1], params[2], wires=sinkbit)
#     qml.Rot(params[3], params[4], params[5], wires=targetbit)
#     qml.CNOT(wires=[sinkbit,targetbit])
#     #qml.Rot(params[6], params[7], params[8], wires=targetbit)

# # General function for applying convolution and pooling
# def apply_layer(params, wires, pool_fn):
#     # Apply convolution on even-indexed qubits
#     for idx in range(0, len(wires) - 1, 2):
#         conv(wires[idx], wires[idx + 1], params[:6])

#     # Apply convolution on odd-indexed qubits
#     for idx in range(1, len(wires) - 1, 2):
#         conv(wires[idx], wires[idx + 1], params[:6])

#     # Apply convolution between last and first qubit
#     conv(wires[-1], wires[0], params[:6])

#     # Apply pooling
#     half_size = len(wires) // 2
#     for idx in range(half_size):
#         pool_fn(wires[idx], wires[idx + half_size], params[6:15]) #non li usa tutti questi parametri dopo dea

# # Define specific layers using the generic function
# def layer_lessZ(params, wires):
#     apply_layer(params, wires, pool_lessZ)

# def layer_lessRot(params, wires):
#     apply_layer(params, wires, pool_lessRot)

# def model_pqc(params, n_qubits):
#     layer_lessZ(params[:14], wires=range(n_qubits))
#     layer_lessRot(params[14:26], wires=[2,3])

@qml.qnode(dev)
def qnode(inputs, weights):
    embedding_VQC(inputs.squeeze())
    # model_pqc(weights.squeeze(), len(inputs.squeeze()))
    apply_layer(weights.squeeze(),  wires=range(len(inputs.squeeze())))

    # qml.BasicEntanglerLayers(weights=inputs, wires=range(4), rotation=qml.RX)
    # qml.BasicEntanglerLayers(weights, wires=range(4))

    # return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    wires = range(len(inputs.squeeze()))
    return [qml.expval(qml.PauliZ(i)) for i in wires if i % 2 == 1]

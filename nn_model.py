import numpy as np
# import pennylane.numpy as np  # PennyLane-compatible NumPy

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_layer import * 


    
#Net type I 
class Net(nn.Module):
    def __init__(self, num_qubits, backend, shift, shots):
        super(Net, self).__init__()

        self.num_qubits = num_qubits
        self.backend = backend
        self.shift = shift 
        self.shots = shots

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        #self.conv2_drop = nn.Dropout2d()

        # self.fc1 = nn.Linear(2304, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 42)

        self.fc4 = nn.Linear(2304, 1*num_qubits)
        
        # self.qc = TorchCircuit.apply
        # self.qc = TorchCircuitWrapper(num_qubits, backend, shift, shots) #eliminate

        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (1, 15)})

        # self.fakeqc = nn.Linear(4, 16)

        # self.fc5 = nn.Linear(16, 10)
        # self.fc5 = nn.Linear(16, 2) 
        # self.fc5 = nn.Linear(4, 2)
        self.fc5 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(-1, 2304)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = F.relu(self.fc2(x))

        #x = F.relu(self.fc3(x))

        x = self.fc4(x)
        x = np.pi*torch.tanh(x)

        # print('input of QC = {}'.format(x))
        # print(x.shape, 'shape')

        # x = self.qc(x[0],  self.num_qubits, self.backend, self.shots, self.shift) # QUANTUM LAYER this is the one to use

        x = self.qlayer(x)
        # print('output of QC = {}'.format(x))
        # print(x.shape, 'shape')

        # x = (x+1)/2

        # x = self.fakeqc(x)
        
#         x = F.relu(x)
        
#         # softmax rather than sigmoid
        # x = self.fc5(x.float())
#         #print('output of Linear(1, 2): {}'.format(x))
        x = F.softmax(x, 1)

        #x = torch.sigmoid(x)
        #x = torch.cat((x, 1-x), -1)
        return x
    
    
    def predict(self, x):
        # apply softmax
        pred = self.forward(x)
#         print(pred)
        ans = torch.argmax(pred[0]).item()
        return torch.tensor(ans)
    


# # QISKIT LAYER

# # class TorchCircuitWrapper:
# #     def __init__(self, num_qubits, backend, shift, shots):
# #         self.num_qubits = num_qubits
# #         self.shift = shift
# #         self.qiskit_circ = QiskitCircuit(num_qubits, backend, shots)
# #         self.qiskit_circ.shift = shift  # Add this so `TorchCircuit` can access it

# #     def __call__(self, x):
# #         return TorchCircuit.apply(x, self.qiskit_circ)
    
# #Net type I 
# class Net(nn.Module):
#     def __init__(self, num_qubits, backend, shift, shots):
#         super(Net, self).__init__()

#         self.num_qubits = num_qubits
#         self.backend = backend
#         self.shift = shift 
#         self.shots = shots

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

#         #self.conv2_drop = nn.Dropout2d()

#         # self.fc1 = nn.Linear(2304, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 42)

#         self.fc4 = nn.Linear(2304, 1*num_qubits)
        
#         self.qc = TorchCircuit.apply
#         # self.qc = TorchCircuitWrapper(num_qubits, backend, shift, shots) #eliminate


#         # self.fakeqc = nn.Linear(4, 16)

#         # self.fc5 = nn.Linear(16, 10)
#         self.fc5 = nn.Linear(16, 2) 

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))

#         x = x.view(-1, 2304)
#         #x = F.relu(self.fc1(x))
#         #x = F.dropout(x, training=self.training)
#         #x = F.relu(self.fc2(x))

#         #x = F.relu(self.fc3(x))

#         x = self.fc4(x)
#         x = np.pi*torch.tanh(x)

#         # print('input of QC = {}'.format(x))
#         # print(x.shape, 'shape')

#         x = self.qc(x[0],  self.num_qubits, self.backend, self.shots, self.shift) # QUANTUM LAYER this is the one to use

#         # x = self.fakeqc(x)
        
#         x = F.relu(x)
#         # print('output of QC = {}'.format(x))
#         # print(x.shape, 'shape')
        
# #         # softmax rather than sigmoid
#         x = self.fc5(x.float())
#         #print('output of Linear(1, 2): {}'.format(x))
#         x = F.softmax(x, 1)

#         #x = torch.sigmoid(x)
#         #x = torch.cat((x, 1-x), -1)
#         return x
    
    
#     def predict(self, x):
#         # apply softmax
#         pred = self.forward(x)
# #         print(pred)
#         ans = torch.argmax(pred[0]).item()
#         return torch.tensor(ans)
    
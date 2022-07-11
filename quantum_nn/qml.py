#Qiskit
from qiskit.circuit import Parameter
from qiskit import execute

# Numpy
import numpy as np
from numpy import pi

# Torch
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Misc
from dataclasses import dataclass
import itertools
import logging
from matplotlib import pyplot as plt
from datetime import datetime
import pathlib
import os

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

#Datetime
dt_string = datetime.now().strftime("[%d.%m.%Y-%H:%M:%S]")

# File Path
file_path = pathlib.Path(__file__).parent.absolute().as_posix()

# -----------------------------------------------------------------------------
# Data Class
# -----------------------------------------------------------------------------

@dataclass
class CircuitParams:
    """Meta parameters needed for building the circuit and neural network"""
    num_qubits: int
    num_thetas: int
    num_shots: int = 5000
    shift: float = pi/4
    learning_rate: float = 0.001
    backend: None = None
    epochs: int = 20
    dataset: str = None
    
# -----------------------------------------------------------------------------
# Quantum Circuit
# -----------------------------------------------------------------------------

class QMLQuantumCircuit():
    """
    Generically building the circuit
    """
    def __init__(self, circuit, circuit_params):
        # --- Circuit definition ---
        print("INIT")
        self.circuit_params = circuit_params
        self.circuit_runs = 1
                
    def N_qubit_expectation_Z(self,counts):
        qc_outputs = self.create_QC_OUTPUTS()
        expects = np.zeros(len(qc_outputs))
        for k in range(len(qc_outputs)):
            key = qc_outputs[k]
            perc = counts.get(key, 0) / self.circuit_params.num_shots
            expects[k] = perc
        return expects
    
    def create_QC_OUTPUTS(self):
        measurements = list(itertools.product([0, 1], repeat=self.circuit_params.num_qubits))
        return [''.join([str(bit) for bit in measurement]) for measurement in measurements]
    
    def run(self, thetas): 
        print(f"Circuit is run: {self.circuit_runs}")
        self.circuit_runs = self.circuit_runs + 1
        job_sim = execute(self.circuit,
                              self.circuit_params.backend,
                              shots = self.circuit_params.num_shots,
                              parameter_binds = [{self.thetas[k] : thetas[k].item() for k in range(self.circuit_params.num_thetas)}])
        result_sim = job_sim.result()
        counts = result_sim.get_counts()
        return self.N_qubit_expectation_Z(counts)
    
    
# Helper function
def generate_thetas(amount):
    return {k : Parameter('Theta'+str(k))for k in range(amount)}

# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------

class HybridFunction(Function):    

    @staticmethod
    def forward(ctx, input, customCircuit):
        if not hasattr(ctx, 'QuantumCircuit'):
            ctx.QuantumCircuit = customCircuit
        if not hasattr(ctx, 'circuit_params'):
            ctx.circuit_params = customCircuit.circuit_params
        exp_value = ctx.QuantumCircuit.run(input)
        result = torch.tensor([exp_value])
        ctx.save_for_backward(result, input)
                
        return result
    
    @staticmethod
    def backward(ctx, grad_output):

        forward_tensor, input = ctx.saved_tensors
        gradients = torch.Tensor()
        for k in range(ctx.circuit_params.num_thetas):
            shift_right = input.detach().clone()
            shift_right[k] = shift_right[k] + ctx.circuit_params.shift
            shift_left = input.detach().clone()
            shift_left[k] = shift_left[k] - ctx.circuit_params.shift
                        
            expectation_right = ctx.QuantumCircuit.run(shift_right)
            expectation_left  = ctx.QuantumCircuit.run(shift_left)
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients = torch.cat((gradients, gradient.float()))
            
        result = torch.Tensor(gradients)
        return (result.float() * grad_output.float()).T, None
    
class NeuralNet(nn.Module):
    def __init__(self, customCircuit):
        self.customCircuit = customCircuit
        
        super(NeuralNet, self).__init__()
        # For grey scale
        self.conv_grey = nn.Conv2d(1, 6, kernel_size=5)
        # For RGB
        self.conv_rgb = nn.Conv2d(3, 6, kernel_size=5)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        # For grey scale
        self.fc_grey = nn.Linear(256, 64)
        # For RGB
        self.fc_rgb = nn.Linear(400, 64)

        self.fc2 = nn.Linear(64, customCircuit.circuit_params.num_thetas)
        self.qc = HybridFunction.apply
        
    def forward(self, x):
        if x.shape[1] == 1:
            x = F.relu(F.max_pool2d(self.conv_grey(x), 2))
        elif x.shape[1] == 3:
            x = F.relu(F.max_pool2d(self.conv_rgb(x), 2))
        else:
            raise ValueError(f'Shape {x.shape[1]} not allowed')
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 256)
        x = x.view(-1, torch.numel(x))
        
        if torch.numel(x) == 256:
            x = F.relu(self.fc_grey(x))
        if torch.numel(x) == 400:
            x = F.relu(self.fc_rgb(x))
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = np.pi*torch.tanh(x)
        x = self.qc(x[0], self.customCircuit)
        x = torch.sigmoid(x)
        x = torch.cat((x, 1-x), -1)
        return x
    
    
    def predict(self, x):
        pred = self.forward(x)
        ans = torch.argmax(pred[0]).item()
        return torch.tensor(ans)
    
# -----------------------------------------------------------------------------
# QMLWrapper
# -----------------------------------------------------------------------------

class QMLWrapper:
    def __init__(self, custom_circuit, train_loader, test_loader, plot_results = True):
        self.custom_circuit = custom_circuit
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.plot_results = plot_results
        self.saves_folder = self.create_folder()
        
        # Logger
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(f'{self.saves_folder}/Logs.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.log_meta_information()
    
    def log_meta_information(self):
        self.logger.info(self.custom_circuit.circuit_params)
        
    def train(self):
        self.logger.debug("Starting training")
        epochs = self.custom_circuit.circuit_params.epochs
        loss_list = []
        loss_func = nn.CrossEntropyLoss()

        self.network = NeuralNet(self.custom_circuit)
        optimizer = optim.Adam(self.network.parameters(), lr=self.custom_circuit.circuit_params.learning_rate)

        count = 0
        for epoch in range(epochs):
            total_loss = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()        
                output = self.network(data)
                loss = loss_func(output, target)
                loss.backward()

                # Optimize the weights
                optimizer.step()
                total_loss.append(loss.item())
                #print(f"Count: {count}")
                count = count + 1
                
            loss_list.append(sum(total_loss)/len(total_loss))
            self.logger.info('Training [{:.0f}%]\tLoss: {:.4f}'.format(
                100. * (epoch + 1) / epochs, loss_list[-1]))
        self.loss_list = loss_list
            
    def plot_loss(self):
        plt.plot(self.loss_list)
        plt.title(f'Loss after n training iterations on \
                  {self.custom_circuit.circuit_params.num_qubits} qubits \
                  with {self.custom_circuit.circuit_params.num_thetas} thetas')
        plt.xlabel('Training Iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.savefig(f'{self.saves_folder}/Loss.jpg')
        if self.plot_results:
            plt.show()
        
    def get_accuracy(self):
        self.logger.debug('Get Accuracy')
        accuracy = 0
        number = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            number +=1
            output = self.network.predict(data).item()
            accuracy += (output == target[0].item())*1
        self.logger.info("Performance on test data is is: {}/{} = {}%".format(accuracy,number,100*accuracy/number))
        
    def plot_sample_predictions(self):
        n_samples_shape = (8, 6)
        count = 0
        fig, axes = plt.subplots(nrows=n_samples_shape[0], ncols=n_samples_shape[1], figsize=(10, 2*n_samples_shape[0]))

        self.network.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if count == n_samples_shape[0]*n_samples_shape[1]:
                    break
                pred = self.network.predict(data).item()

                axes[count//n_samples_shape[1]][count%n_samples_shape[1]].imshow(data[0].numpy().squeeze(), cmap='gray')

                axes[count//n_samples_shape[1]][count%n_samples_shape[1]].set_xticks([])
                axes[count//n_samples_shape[1]][count%n_samples_shape[1]].set_yticks([])
                axes[count//n_samples_shape[1]][count%n_samples_shape[1]].set_title('Predicted {}'.format(pred))
                
                count += 1
        plt.savefig(f'{self.saves_folder}/SamplePredictions.jpg')
        if self.plot_results:
            plt.show()
        
    def save_circuit(self):
        self.custom_circuit.circuit.draw(output='mpl', filename=f'{self.saves_folder}/circuit.jpg')
        
    def create_folder(self):
        file_path_appendix = f"/Saves\
/{self.custom_circuit.circuit_params.dataset.name}_{self.custom_circuit.circuit_params.num_qubits}-Qubits\
_{self.custom_circuit.circuit_params.num_thetas}-Thetas\
_SIMULATED/{dt_string}"
        os.makedirs(file_path + file_path_appendix)
        return file_path + file_path_appendix
        
        
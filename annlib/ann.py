import os, time, pickle
import lib.config as config
from lib.layer import *
from lib.link import *
from lib.backprop import *


# Artificial Neural Network
#
# Usage:
# ann = ANN(layers, links, mode) 
#  => Create new ANN
# ann.eval(input_data) 
#   => Evaluate input data
# ann.train(input_data, output_data) 
#   => Train network with inputs and output, with specified mode
# ann.train_set(input_data_set, output_data_set, max_epochs, error_threshold)
#   => Train a set of inputs and outputs, with optional max epoch count 
#      and error threshold

class ANN:
    
    def __init__(self, layers, links, mode = 'training-backpropagation'):
        self.layers = layers
        self.links = links
        self.mode = mode
        self.backpropagator = Backprop(self)
    
    # Set mode of the network
    def set_mode(self, mode):
        self.mode = mode
    
    # Evaluate array of input data
    def eval(self, input_data):
        self.feedforward(input_data)
        return self.outputs()
    
    # Train with set of arrays of input_data and output_data,
    # run to max_epochs or until error threshold is reached.
    def train_set(self, input_set, output_set, max_epochs = 10**5, threshold = 0.1**5):
        for epoch in range(max_epochs):
            if epoch % 100 == 0: print 'epoch', epoch
            error = 0
            for i in range(len(input_set)):
                error += self.train(input_set[i], output_set[i])
            error /= len(input_set)
            if threshold and error < threshold: break
    
    # Train with array of input and output data
    def train(self, input_data, output_data):
        self.eval(input_data)
        if self.mode == 'training-backpropagation':
            self.backprop(output_data)
        return self.error(output_data)
    
    # Run input data through network
    def feedforward(self, input_data):
        self.reset_layers()
        self.layers[0].update_nodes(self.mode, input_data)
        for i in range(1, len(self.layers)):
            if config.display: print "\nUpdating layer " + self.layers[i].options['name'] + "\n"
            self.layers[i].update_nodes(self.mode)
            if self.mode == 'training-unsupervised':
                self.layers[i].train_hebbian()
            if config.display: 
                for layer in self.layers: layer.draw()
        if config.display: print "-" * 60
    
    # Adjust weights through backpropagation
    def backprop(self, output_data):
        self.backpropagator.compute_deltas(output_data)
        self.backpropagator.update_weights()
    
    # Calculate error between current network output and given output data
    def error(self, output_data):
        error = []
        for i,o in zip(output_data, self.outputs()):
            error.append( math.fabs(i-o) )
        return sum(error) / len(error)
    
    # Current network output data
    def outputs(self):
        return self.layers[-1].outputs()
    
    # Reset node activation levels and backprop error terms
    def reset_layers(self):
        for layer in self.layers: 
            layer.reset_activation_levels()
            layer.reset_error_terms()
    
    # Load serialized network
    @staticmethod
    def load(name):
        print 'Loading', name + '.ann'
        with open(config.base_dir + 'networks/' + str(name) + '.ann', 'rb') as f:
            return pickle.load(f)
    
    # Save network by serialization
    def save(self, name):
        print 'Saving', name + '.ann'
        with open(config.base_dir + 'networks/' + str(name) + '.ann', 'wb') as f:
            pickle.dump(self, f)

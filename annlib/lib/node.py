import math

class Node:
    
    def __init__(self, layer):
        self.layer = layer
        self.membrane_potential = 0.0
        self.activation_level = 1.0
        self.prev_activation_level = 1.0
        self.activation_level_derivative = 0.0 # for backpropagation
        self.reset = True # for possible future use with leaking activation
        self.error_term = 0.0 # for backpropagation
        self.is_bias = False
    
    def set_bias(self):
        self.is_bias = True
        self.activation_level = 1.0
        self.prev_activation_level = 1.0
        
    def update(self, net):
        if self.is_bias:
            self.activation_level = 1.0
            self.activation_level_derivative = 0.0
        else:
            self.membrane_potential += net
            self.activation_level = self.activation()
            self.activation_level_derivative = self.derivation()
            if self.reset: self.membrane_potential = 0.0
    
    def update_prev_activation(self):
        self.prev_activation_level = self.activation_level
    
    def delta(self):
        return math.fabs(self.activation_level - self.prev_activation_level)

    def reset_activation_level(self):
        self.prev_activation_level = 0.0
        self.activation_level_derivative = 0.0
        self.activation_level = 0.0

    def reset_error_term(self):
        self.error_term = 0.0

    ## Activation functions
    
    def activation(self):
        f = self.layer.options['activation_function']
        return getattr(self, f + '_activation')()
    
    # Hopfield neural net activation
    def hopfield_activation(self):
        if self.membrane_potential > 0.0: return 1.0
        elif self.membrane_potential == 0.0: return 0.0
        else: return -1.0
    
    # A simple linear function that outputs the sum of weighted inputs 
    # (without transforming that sum in any way).
    def linear_activation(self):
        return self.membrane_potential
    
    # A positive linear function that outputs the sum of weighted inputs when that sum is positive; 
    # otherwise it outputs a 0.
    def positive_linear_activation(self):
        m = self.membrane_potential
        return m if m > 0.0 else 0.0
    
    # A step function with a threshold, T, that outputs values in the range [0,1]. 
    # Standard values for T include 0, 0.5 and 1, depending upon the situation.
    def step_activation(self):
        if self.membrane_potential > self.layer.options['step_threshold']:
            return 1.0
        else:
            return 0.0
    
    # A sigmoid logistic function, which outputs values in the range [0,1].
    def logistic_activation(self):
        return 1.0 / (1.0 + math.exp(-1.0 * self.membrane_potential))
    
    # A sigmoid tanh function, which outputs values in the range [-1, 1].
    def tanh_activation(self):
        return math.tanh(self.membrane_potential)
    
    
    ## Derivation of activation functions
    
    def derivation(self):
        f = self.layer.options['activation_function']
        try: return getattr(self, f + '_derivation')()
        except: raise Exception('derivation of function not implemented: ' + f)
    
    def linear_derivation(self):
        return 1.0
    
    def logistic_derivation(self):
        return self.activation_level * (1 - self.activation_level)
    
    def tanh_derivation(self):
        return 1 - self.activation_level ** 2
        
    
    def hopfield_derivation(self):
        return 0
    
    def step_derivation(self):
        return 0
    

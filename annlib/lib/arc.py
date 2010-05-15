import math

class Arc:
  
    def __init__(self, link, pre, post, initial_weight = 0.0):
        self.pre = pre
        self.post = post
        self.link = link
        self.initial_weight = initial_weight
        self.weight = initial_weight
        self.deltas = [] # for backpropagation
        self.prev_delta = 0.0 # for backpropagation
    
    def set_initial_weight(self, val):
        self.weight = val
        self.initial_weight = val
    
    def reset_weight(self):
        self.weight = self.initial_weight
    
    def weighted_activation(self):
        return self.pre.activation_level * self.weight
    
    def weighted_prev_activation(self):
        return self.pre.prev_activation_level * self.weight
    
    def adjust_weight(self):
        self.weight = self.weight + self.weight_change()
    
    # Find the weight change based on learning rule
    def weight_change(self):
        try: return getattr(self, self.link.options['learning_rule'] + '_rule')()
        except: raise Exception('learning rule not implemented: ' + self.link.options['learning_rule'])
    
    def learning_rate(self):
        return self.link.options['learning_rate']
    
    # Find the arc weight change with the classic hebbian rule
    def classic_hebb_rule(self):
        pre = self.pre.activation_level
        post = self.post.activation_level
        return self.learning_rate() * pre * post
    
    # Find the arc weight change with the general hebb rule
    def general_hebb_rule(self):
        pre = self.pre.activation_level
        post = self.post.activation_level
        theta = self.link.options['theta']
        return self.learning_rate() * (pre - theta) * (post - theta)
    
    # Find the arc weight change with the oja rule
    def oja_rule(self):
        pre = self.pre.activation_level
        post = self.post.activation_level
        return self.learning_rate() * ((pre * post) - ((post ** 2) * math.fabs(self.weight)))

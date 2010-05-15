import random, yaml, utils, re, config
from arc import *

class Link:
    
    def __init__(self, options = {}):
        defaults = yaml.load(file(config.base_dir + 'scripts/defaults.yml', 'r'), Loader=yaml.Loader)['link']
        self.options = utils.parse_options(options, defaults)
    
    # Total input to node
    def get_net_inputs_to_node(self, node):
        post_node_index = self.post.nodes.index(node)
        arcs = self.arcs[post_node_index]
        net = 0.0
        for arc in arcs:
            if not arc: continue
            if self.pre == self.post:
                net += arc.weighted_prev_activation()
            else:
                net += arc.weighted_activation()
        return net
    
    # Adjust the weights of all arcs
    def adjust_all_weights(self):
        if not self.options['plastic']: return
        for i in range(len(self.arcs)):
            for j in range(len(self.arcs[0])):
                if self.arcs[i][j]: self.arcs[i][j].adjust_weight()
    
    # All arcs in 1D array
    def all_arcs(self):
        arcs = []
        for row in self.arcs: 
            for arc in row:
                if arc: arcs.append(arc)
        return arcs
    
    ## Generator
    
    def generate_arcs(self):
        self.arcs = [ [False for col in range(len(self.pre.nodes))] for row in range(len(self.post.nodes)) ]

    # Create a new arc at position [i,j]
    def new_arc(self, i, j):
        self.arcs[i][j] = Arc(self, self.pre.nodes[j], self.post.nodes[i], self.initial_weight())
    
    # Get initial weight for arc
    def initial_weight(self):
        w = self.options['weights']
        if isinstance(w, list):
            return random.uniform(w[0], w[1])
        else:
            return w


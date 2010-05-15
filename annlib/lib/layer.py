import math, yaml, config, utils
from node import *

class Layer:
  
    def __init__(self, options = {}):
        defaults = yaml.load(file(config.base_dir + 'scripts/defaults.yml', 'r'), Loader=yaml.Loader)['layer']
        self.options = utils.parse_options(options, defaults)
        self.nodes = []
        self.links_in = []
        self.links_out = []
        self.is_active = True
        for i in range(self.options['size']): 
            self.nodes.append(Node(self))
        
        # Add bias node
        if self.options['bias_node']:
            self.nodes.append(Node(self))
            self.nodes[-1].set_bias()
        
    
    ## Execution
    
    def recurring_link(self):
        for link in self.links_out:
            if link.post == self: return True
        return False
    
    # Sanity check for incoming data
    def check_inputs(self, data):
        if data and self.options['size'] != len(data):
            raise Exception("Non-matching data input and number of nodes.")
    
    # Check if node values have converged
    def quiescent(self):
        for node in self.nodes:
            if node.delta() > self.options['quiescent_threshold']:
                return False
        return True
    
    # Update all nodes
    def update_nodes(self, mode, data = False):
        mode = mode.split('-')[0]
        if not self.options[mode]['is_active']: return
        if data: self.check_inputs(data)
        
        if self.options[mode]['quiescence'] and self.recurring_link():
            print "Q:",
            self.draw_layer()
            for i in range(max(1, self.options['max_rounds'])):
                self.update_nodes_once()
                print "Q:",
                self.draw_layer()
                if self.quiescent(): break
            print
        else:
            self.update_nodes_once(data)
    
    # Update all nodes once
    def update_nodes_once(self, data = False):
        for node in self.nodes:
            node.update_prev_activation()
        if data:
            for i in range(len(data)):
                self.nodes[i].update(data[i])
            if self.options['bias_node']:
                self.nodes[-1].update(0)
        else:
            for i in range(len(self.nodes)):
                net = 0.0
                for link in self.links_in:
                    net += link.get_net_inputs_to_node(self.nodes[i])
                self.nodes[i].update(net)
    
    # Train all link weights, hebb style
    def train_hebbian(self):
        for link in self.links_in: link.adjust_all_weights()
    
    # Output values for last layer
    def outputs(self):
        return [node.activation_level for node in self.nodes]
    
    def reset_activation_levels(self):
        for node in self.nodes: node.reset_activation_level()

    def reset_error_terms(self):
        for node in self.nodes: node.reset_error_term()
    

    ## Drawing
    
    # Display layer and its links
    def draw(self):
        self.draw_layer()
        print
        self.draw_links()
    
    def draw_layer(self):
        print utils.string_to_exact_len(self.options['name'], 12),
        print ' | ',
        for node in self.nodes:
            print utils.string_to_exact_len(round(node.activation_level, 2), 20),
            print ' | ',
        print
    
    def draw_links(self):
        for link in self.links_out:
            if not link.options['display']: continue
            print utils.string_to_exact_len(link.options['name'], 13),
            for arcrow in link.arcs:
                if not arcrow[0]: continue
                print ' | ',
                for arc in arcrow:
                    if arc: print utils.string_to_exact_len(round(arc.weight, 2), 5),
                    else: print ' ' * 5,
                    print ' | ',
                print
                print ' ' * 13,
            print "\n"            

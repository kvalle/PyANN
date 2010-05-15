class Backprop:
    
    momentum = 0.1
    online = False
    
    def __init__(self, ann):
        self.ann = ann
    
    def compute_deltas(self, solution):
        output_layer = self.ann.layers[-1]
        internal_layers = self.ann.layers[1:-1]
        
        # compute error terms for output layer
        for i in range(len(output_layer.nodes)):
            node = output_layer.nodes[i]
            out  = node.activation_level
            node.error_term = (solution[i] - out) * node.activation_level_derivative
            if self.online: self.update_arcs_in_layer(output_layer)
            
        # compute error terms for internal layers
        for layer in reversed(internal_layers):
            # error terms for each node in layer
            for node in layer.nodes:
                error = 0.0
                for arc in self.arcs_from_node(node):
                    error += arc.weight * arc.post.error_term
                node.error_term = error * node.activation_level_derivative
            if self.online: self.update_arcs_in_layer(layer)
        
        if not self.online: self.append_deltas()
    
    def append_deltas(self):
        # add weight deltas
        for link in self.ann.links:
            rate = link.options['learning_rate']
            for arc in link.all_arcs():
                delta = rate * arc.post.error_term * arc.pre.activation_level
                arc.deltas.append(delta)
    
    def update_weights(self):
        if self.online: return
        for link in self.ann.links:
            for arc in link.all_arcs():
                self.momentum *= 0.995
                s = sum(arc.deltas)
                arc.weight += s + self.momentum * arc.prev_delta
                arc.prev_delta = s
                arc.deltas = []
    
    def update_arcs_in_layer(self, layer):
        for link in layer.links_in:
            for arc in link.all_arcs():
                rate = link.options['learning_rate']
                delta = rate * arc.post.error_term * arc.pre.activation_level
                arc.weight += delta
    
    def arcs_from_node(self, node):
        links = node.layer.links_out
        arcs = []
        for link in links:
            for arc in link.all_arcs():
                if arc.pre == node:
                    arcs.append(arc)
        return arcs

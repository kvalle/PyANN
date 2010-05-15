import yaml
import lib.config as config
from ann import *
from case import *
from lib.layer import *
from lib.link import *

class Generator:
    
    @staticmethod
    def generate_ann(case):
        return Generator().generate(case.casefile['script'])
    
    # Generate ANN from self.script
    def generate(self, script):
        self.script = yaml.load(file(config.base_dir + 'scripts/' + script + '.yml', 'r'), Loader=yaml.Loader)
        
        layers = self.generate_layers()
        links  = self.generate_links()
        
        self.connect_links_and_layers(layers, links)
        self.generate_arcs(links)
        ordered_layers = self.order_layers(layers)
        
        ann = ANN(ordered_layers, links.values())
        return ann
        
    
    # Create dictionary of layer objects from script
    def generate_layers(self):
        layers = {}
        for name in self.script['layers']:
            cfg = self.script['layers'][name]
            cfg['name'] = name
            layers[name] = Layer(cfg)
        return layers    
    
    # Create dictionary of link objects from script
    def generate_links(self):
        links = {}
        for name in self.script['links']:
            cfg = self.script['links'][name]
            cfg['name'] = name
            links[name] = Link(cfg)
        return links
    
    # Tell links to generate arches
    def generate_arcs(self, links):
        for link in links.values(): 
            link.generate_arcs()
            getattr(self, 'new_' + link.options['connection_type'] + '_connection')(link)
            self.remove_arcs_to_bias(link)
    
    # Set up connections between links and layers from script
    def connect_links_and_layers(self, layers, links):
        cfg = self.script['links']
        for name in cfg:
            # add links to layers
            layers[cfg[name]['pre']].links_out.append(links[name])
            layers[cfg[name]['post']].links_in.append(links[name])        
            # add layers to links
            links[name].pre = layers[cfg[name]['pre']]
            links[name].post = layers[cfg[name]['post']]
    
    # Create an ordered array of layers from script ordering and layer dict
    def order_layers(self, layers):
        order = self.script['order']
        ordered_layers = []
        for name in order:
            ordered_layers.append(layers[name])
        return ordered_layers



    ## Arc generators
    
    # Create a new full connection set of link.arcs
    def new_full_connection(self, link):
        for i in range(len(link.arcs)):
            for j in range(len(link.arcs[0])):
                link.new_arc(i, j)

    # Create a new 1-1 connection set of link.arcs
    def new_one_to_one_connection(self, link):
        for i in range(0, min(len(link.pre.nodes), len(link.post.nodes))):
            link.new_arc(i, i)    

    # Create a new stochastic connection set of link.arcs
    def new_stochastic_connection(self, link):
        for i in range(len(link.arcs)):
            for j in range(len(link.arcs[0])):
                if self.options['stochastic_p'] < random.random():
                    link.new_arc(i, j)

    # Create a new triangular connection set of link.arcs
    def new_triangular_connection(self, link):
        for i in range(len(link.arcs)):
            for j in range(len(link.arcs[0])):
                if i != j: link.new_arc(i, j)            

    # Create a new custom connection from settings
    def new_custom_connection(self, link):
        custom = [ [False for col in range(len(link.post.nodes))] for row in range(len(link.pre.nodes)) ]
        for i in range(len(link.options['custom'])):
            row = re.split('[ ]+', link.options['custom'][i])
            for j in range(len(row)):
                if row[j] != 'x' and row[j] != '':
                    custom[i][j] = float(row[j])
                    
        for i in range(len(link.arcs)):
            for j in range(len(link.arcs[0])):
                if custom[j][i]:
                    link.new_arc(i,j)
                    link.arcs[i][j].set_initial_weight(custom[j][i])
    
    def remove_arcs_to_bias(self, link):
        for i in range(len(link.arcs)):
            for j in range(len(link.arcs[0])):
                if link.arcs[i][j] and link.arcs[i][j].post.is_bias:
                    link.arcs[i][j] = False
        
    
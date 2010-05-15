import annlib.lib.config as config
import annlib.generator as generator
import annlib.case as case

# 
# Use this file to load cases, create and train networks.
# 
# * Create a casefile in the 'cases' folder
# * Use a script from the 'scripts' folder
# * The finished network is saved in the 'networks' folder.
# 

# Select case
casename = 'avoidance'

###########################################################

# Load case
print 'Loading case', casename
case = case.Case(casename)

# Directory of the ANN library
config.base_dir = 'annlib/'

# Generate case ANN
print 'Generating ANN for', casename
ann = generator.Generator.generate_ann(case)

# Run the case
print 'Running case', casename
output = case.run(ann)

# Report testing results
if case.tasks and case.tasks[-1].is_testing():
    print 'Testing network:'
    print 'given input', case.tasks[-1].input()
    print 'expected output', case.tasks[-1].output()
    print 'actual output', output

# Display weights
# for layer in ann.layers:
#     layer.draw_links()

# Save network
#ann.save(casename)

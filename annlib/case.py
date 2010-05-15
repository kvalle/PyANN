import yaml
import math
import lib.config as config

class Case:
    
    def __init__(self, name):
        try: self.casefile = yaml.load(file(config.base_dir + 'cases/' + name + '.yml', 'r'), Loader=yaml.Loader)
        except: raise Exception("Unknown case: " + name)
        self.tasks = []
        if 'tasks' in self.casefile:
            for task in self.casefile['tasks']: self.add_task(self.__parse_text_task(task))
    
    def add_task(self, task):
        self.tasks.append(Task(task))
    
    def clear_tasks(self):
        self.tasks = []
    
    def __parse_text_task(self, task):
        if 'input' in task:
            task['input'] = self.__parse_data(task['input'])
        if 'output' in task:
            task['output'] = self.__parse_data(task['output'])
        return task
        
    def __parse_data(self, data):
        if isinstance(data, str):
            datas = self.__strings_to_floats(data)
        else:
            datas = []
            for group in data:
                datas.append(self.__strings_to_floats(group))
        return datas
    
    def __strings_to_floats(self, l):
        new = []
        for c in l.split(' '): new.append(float(c))
        return new
    
    def run(self, ann):
        for task in self.tasks:
            ann.set_mode(task.mode())
            if task.mode() == 'testing':
                outputs = []
                for inputs in task.input():
                    outputs.append( ann.eval(inputs) )
                return outputs
            else:
                ann.train_set(task.input(), task.output(), task.epochs(), task.error_threshold())
    

class Task:
    
    def __init__(self, task):
        self.task = task
    
    def input(self):
        return self.task['input']
        
    def output(self):
        return self.task['output']
        
    def epochs(self):
        if 'epochs' in self.task: return self.task['epochs']
        else: return 1
    
    def error_threshold(self):
        if 'error_threshold' in self.task: return self.task['error_threshold']
        else: return 0.000001
    
    def mode(self):
        if 'mode' in self.task: return self.task['mode']
        else: return 'testing'
    
    def momentum(self):
        if 'momentum' in self.task: return self.task['momentum']
        else: return 0
    
    def is_testing(self):
        return self.mode().split('-')[0] == 'testing'

import json
import copy
import os
import random
random.seed(42)
data_path = os.path.join(os.path.dirname(__file__), "settings_sample.json")

class HashableDict(dict):
    def __hash__(self):
        return str(self).__hash__()

def cartesian_prod(grids, name=''):
    keys = list(grids.keys())
    res = set()
    def dfs(fruit, depth = 0):
        if depth == len(keys):
            res.add(fruit)
            return
        key = keys[depth]

        for opt in grids[key]:
            newfruit = copy.deepcopy(fruit)
            newfruit[key] = opt
            dfs(newfruit, depth+1)
    dfs(HashableDict())
    return res

class ExpVariable:
    def __init__(self, data, name='root'):
        self.name = name
        self.data = data
        self.vars = {}
        self.opt_groups = {}
        self.parse()
        if self.name == 'root':
            self.node_node = list(self.opt_groups["node_node"])
            self.node_node.sort(key=lambda x: str(x))
            self.node_graph = list(self.opt_groups["node_graph"])
            self.node_graph.sort(key=lambda x: str(x))
            self.node_list = self.node_node + self.node_graph
            self.graph_graph = list(self.opt_groups["graph_graph"])
            self.graph_graph.sort(key=lambda x: str(x))
            self.graph_node = list(self.opt_groups["graph_node"])
            self.graph_node.sort(key=lambda x: str(x))
            self.graph_list = self.graph_graph + self.graph_node        
            self.split = {}
            self.sample()

    def all(self):
        ret = set()
        for k, v in self.opt_groups.items():
            #print("--all ", k)
            ret = ret.union(v)
        return ret

    def sample(self, s=100):
        self.split['node_enc'] = sample_exps(self.node_list, 'enc', 'gcn', s) 
        self.split['node_dec'] = sample_exps(self.node_list, 'dec', 'inner', s)
        self.split['node_est'] = sample_exps(self.node_list, 'est', 'jsd', s)
        self.split['node_sampler'] = sample_exps(self.node_list, 'sampler', 'dgi', s)
        self.split['node_readout'] = sample_exps(self.node_graph, 'readout', 'mean', s)
        self.split['graph_enc'] = sample_exps(self.graph_list, 'enc', 'gcn', s)
        self.split['graph_dec'] = sample_exps(self.graph_list, 'dec', 'inner', s)
        self.split['graph_est'] = sample_exps(self.graph_list, 'est', 'jsd', s)
        self.split['graph_sampler'] = sample_exps(self.graph_list, 'sampler', 'dgi', s)
        self.split['graph_readout'] = sample_exps(self.graph_graph, 'readout', 'mean', s)

    def sample_single(self, task, module, ins):
        if module == 'readout':
            return sample_exps(self.node_graph, module, ins)
        

    def select(self, set_of_opt_groups):
        ret = set()
        for k, v in self.opt_groups.items():
            if k in set_of_opt_groups:
                #print("--select ", v)
                ret = ret.union(v)
        return ret

    def parse_options(self, data):
        for k, v in data.items():
            self.opt_groups[k] = set()
            if isinstance(v, list):  # a simple opt group
                self.opt_groups[k] = self.opt_groups[k].union(set(v))
            else: # a generation rule
                # parse this rule
                grids = {}
                for var, grps in v.items():  # select groups of var
                    if grps == 'all':
                        grids[var] = self.vars[var].all()
                    else:  # selected groups
                        grids[var] = self.vars[var].select(set(grps))
                for var in self.vars.keys():  # omitted
                    if var not in grids:
                        grids[var] = self.vars[var].all()
                self.opt_groups[k] = cartesian_prod(grids, k)



    def parse(self):
        for k, v in self.data.items():
            if k == '_options':  # options
                self.parse_options(v)
                continue
            # variable
            self.vars[k] = ExpVariable(v, name=k)
        # print(self)

    def __str__(self):
        space = '\n        '
        s = '\n\n'.join([
            f"name:   {self.name}",
            f"vars:   {space.join(self.vars.keys())}",
            f"opts:   {space.join([k + ': ' + str(len(self.opt_groups[k])) for k in self.opt_groups.keys()])}"
        ])
        return '------VAR-------\n' + s + '\n-------VAR-------\n'

def parse():
    with open(file=data_path, mode='r') as fp:
        data = json.load(fp)
        ret = ExpVariable(data)
        return ret

def gen_bash(opt):
    s = "python3 -m opengcl"
    def dfs(opt):
        res = ''
        keys = list(opt.keys())
        keys.sort()
        for k in keys:
            v = opt[k]
            if k.startswith('_'):
                if isinstance(v, dict):
                    res += dfs(v)
            else:
                res += ' --' + k + ' ' + v

        return res

    s += dfs(opt) + " $*"
    return s

def sample_exps(exps, k ,v, size=100):
    random.shuffle(exps)
    sample_dict = {
        'enc': ['none', 'linear', 'gcn', 'gat', 'gin'],
        'dec': ['inner', 'bilinear'],
        'sampler': ["snr",
                    "srr",
                    "gca","dgi", "mvgrl", "graphcl"],
        'readout': ['mean', 'sum'],
        'est': ['jsd', 'nce']
    }
    sampled = []
    id = 0
    for i in exps:
        if i['_model'][k] == v:
            id += 1
            sampled.append(i)
            for p in sample_dict[k]:
                ci = copy.deepcopy(i)
                ci['_model'][k] = p
                sampled.append(ci)
        if id == size:
            break

    return sampled

    


if __name__ == "__main__":
    exps = parse()
    print(exps)
    node_node = list(exps.opt_groups["node_node"])
    node_graph = list(exps.opt_groups["node_graph"])
    node_list = node_node + node_graph
    graph_graph = list(exps.opt_groups["graph_graph"])
    graph_node = list(exps.opt_groups["graph_node"])
    graph_list = graph_graph + graph_node
    s = 100
    sample_node = sample_exps(node_list, 'enc', 'linear', s) + sample_exps(node_list, 'dec', 'inner', s) + sample_exps(node_list, 'est', 'jsd', s) + sample_exps(node_list, 'sampler', 'dgi', s) + sample_exps(node_graph, 'readout', 'mean', s)
    sample_graph = sample_exps(graph_list, 'enc', 'linear', s) + sample_exps(graph_list, 'dec', 'inner', s) + sample_exps(graph_list, 'est', 'jsd', s) + sample_exps(graph_list, 'sampler', 'dgi', s) + sample_exps(graph_graph, 'readout', 'mean', s)
    #print(sample)
    sample_all = list(set(sample_node + sample_graph))
    exps.sampled = sample_all
    for i in sample_all:
        print(str(i))
        print(gen_bash(i))
        break

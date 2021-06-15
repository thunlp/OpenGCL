import pandas as pd
import sys
import os
from sys import argv

curdir = os.path.dirname(__file__)
sys.path.extend([os.path.join(curdir, '')])

exp_name = 'graph_readout'
script_name = os.path.join(curdir, '../../src/autogen_sample_script_' + exp_name + '.sh')



try:
    from . import parse_exps_sampled
    from parse_exps_sampled import ExpVariable
except Exception as _:
    import parse_exps_sampled
    from parse_exps_sampled import ExpVariable
import numpy as np


def gen_work_table(exps: ExpVariable, exp_run, exp_name):
    datasets = list(exps.vars['dataset'].select('unigraph')) + list(exps.vars['dataset'].select('multigraph'))
    table = [[]]
    table[0] = ['', '', '', '', '', ''] + datasets
    attrs = ['enc', 'dec', 'sampler', 'readout', 'est']
    hyperattrs = ['epochs', 'early-stopping', 'learning-rate', 'output-dim', 'hidden-size']
    hyperattrs1 = ['epochs', 'early_stopping', 'learning-rate', 'output-dim', 'hidden-size']
    modelset = {}

    def parse_exp(exp):
        model = exp['_model']
        dataset = exp['dataset']
        hyper = exp['_hyperparameters']
        modelstr = ','.join([optname] + [model[attr] for attr in attrs])
        hyperstr = ','.join([hyper[hyp] for hyp in hyperattrs])
        return model, dataset, hyper, modelstr, hyperstr

    # run 1 -- framework of exp. table, get menu of experiment plan
    split_exp = exps.split[exp_name]
    exp_plan = {parse_exps_sampled.gen_bash(k): {"done": False, "priority": 0} for k in split_exp}
    #print(exp_plan)
    for optname, i in exps.opt_groups.items():
        print("optname", optname)
        if optname[-1].isdigit():
       #     print(" digit", optname[-1])
            for k in exps.opt_groups[optname]:
                kstr = parse_exps_sampled.gen_bash(k)
                if kstr in exp_plan.keys():
                    exp_plan[kstr]["priority"] = int(optname[-1])
       #         if optname[-1] == "1":
       #             print(parse_exps.gen_bash(k))
            continue

        for exp in i:
            model, dataset, hyper, modelstr, hyperstr = parse_exp(exp)
            if modelstr in modelset:
                if dataset in modelset[modelstr]:
                    modelset[modelstr][dataset][hyperstr] = 1
                else:
                    modelset[modelstr][dataset] = {hyperstr: 1}

            else:
                modelset[modelstr] = {dataset: {hyperstr: 1}}


    # run 2 -- go through exp plan again and finish table
    for optname, i in exps.opt_groups.items():
        if optname[-1].isdigit():
            continue
        for exp in i:
            model, dataset, hyper, modelstr, hyperstr = parse_exp(exp)
            if modelset[modelstr][dataset][hyperstr] == 3:
                kstr = parse_exps_sampled.gen_bash(exp)
                if kstr in exp_plan.keys():
                    exp_plan[kstr]['done'] = True

    l1 = [(k, v) for k, v in modelset.items()]
    l1.sort(key=lambda x: x[0])
    for (k, v) in l1:  # k: name of model str; v: {all datasets: apperance}
        item = k.split(',')
        for ds in datasets:
            if ds in v:
                a, b = 0, 0
                for hypers in v[ds]:
                    if v[ds][hypers] == 3:
                        a += 1
                    b += 1
                item.append(str(a) + '/' + str(b))
            else:
                item.append('')
        table.append(item)
    table = np.array(table, dtype='str')

    return table, exp_plan




def generate_exp_data():

    # get datasets

    exps = parse_exps_sampled.parse()

    # generate work-table
    table, exp_plan = gen_work_table(exps, pd.DataFrame(), exp_name)


    if len(argv) == 3 and argv[2] == 'minimal':
        exp_plan = [(k, v) for (k, v) in exp_plan.items() if v['priority'] == 1]
    else:
        exp_plan = [(k, v) for (k, v) in exp_plan.items()]
    all_exps = len(exp_plan)
    exp_plan = [(k, v) for (k, v) in exp_plan if not v['done']]
    new_exps = len(exp_plan)
    print(f"Experiment progress = {all_exps-new_exps}/{all_exps}")

    exp_plan.sort(key=lambda x: -x[1]['priority'])
    with open(script_name, 'w') as fp:
        for (k, v) in exp_plan:
            print(k, file=fp)



if __name__ == '__main__':
    generate_exp_data()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast 

from collections import defaultdict

plt.figure(dpi=600)

def plot(metrics):
    steps = [step for step in range(0, 10*len(metrics['f1']), 10)]
    metric_names = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'}
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    num = 0
    for name, values in metrics.items():
        labell = f"{metric_names[name]} ({round(max(values), 2)})"
        plt.plot(steps, values, marker='', color=palette(num), linewidth=2, alpha=1, label=labell)
        num += 1
    
    # Add legend
    plt.legend(loc=4, ncol=1)

    # Add titles
    plt.xlabel("steps")
    plt.ylabel("score")
    # plt.show()
    print("> Saved figure as results.png")
    plt.savefig('results.png')


def plot_loss(filepath):
    f = open(filepath, "r")
    loss = defaultdict(list)
    for line in f:
        info = ast.literal_eval(line)
        if 'loss' in info:
            loss['Training Loss'].append(info['loss'])
        else:
            loss['Validation Loss'].append(info['eval_loss'])
    
    steps = [step for step in range(0, 10*len(loss['Training Loss']), 10)]
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    num = 0
    for loss_type, values in loss.items():
        plt.plot(steps, values, marker='', color=palette(num), linewidth=2, alpha=1, label=loss_type)
        num += 1
    
    # Add legend
    plt.legend(loc=1, ncol=1)

    # Add titles
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    # plt.show()
    print("> Saved figure as loss.png")
    plt.savefig('loss.png')
    
    

plot_loss('./loss.txt')

# metrics = {'precision': [0.12456633380924724, 0.08321345681185499, 0.036231884057971016, 0.043478260869565216, 0.043478260869565216, 0.043478260869565216, 0.06521739130434782, 0.1068840579710145, 0.26400234226321184, 0.40878244139113695], 
#             'recall': [0.08749677796696499, 0.043985014561677555, 0.036231884057971016, 0.043478260869565216, 0.043478260869565216, 0.043478260869565216, 0.05507246376811594, 0.07504224078436972, 0.1842577175381306, 0.31635212758418974], 
#             'f1': [0.08709705098058107, 0.048872708111838546, 0.036231884057971016, 0.043478260869565216, 0.043478260869565216, 0.043478260869565216, 0.05857487922705313, 0.08269622449853348, 0.20222127405459506, 0.3367030600202325]
#         }

# plot(metrics)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast 
from collections import defaultdict

plt.figure(dpi=600)

def plot(metrics, filename):
    steps = [step for step in range(0, 10*len(metrics['f1']), 10)]
    metric_names = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'}
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    num = 0
    for name, values in metrics.items():
        plt.plot(steps, values, marker='', color=palette(num), linewidth=2, alpha=1, label=f"{metric_names[name]}")
        num += 1
    
    # Add legend at bottom right corner
    plt.legend(loc=4, ncol=1)

    # Add titles
    plt.xlabel("Steps")
    plt.ylabel("Score")
    print(f"> Saved figure as {filename}")
    plt.savefig(filename)


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
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    plt.legend(loc=2, ncol=1)

    # Add titles
    plt.xlabel("steps")
    plt.ylabel("score")
    # plt.show()
    print("> Saved figure as results.png")
    plt.savefig('results.png', dp1=1200)





# metrics = {'precision': [0.12456633380924724, 0.08321345681185499, 0.036231884057971016, 0.043478260869565216, 0.043478260869565216, 0.043478260869565216, 0.06521739130434782, 0.1068840579710145, 0.26400234226321184, 0.40878244139113695], 
#             'recall': [0.08749677796696499, 0.043985014561677555, 0.036231884057971016, 0.043478260869565216, 0.043478260869565216, 0.043478260869565216, 0.05507246376811594, 0.07504224078436972, 0.1842577175381306, 0.31635212758418974], 
#             'f1': [0.08709705098058107, 0.048872708111838546, 0.036231884057971016, 0.043478260869565216, 0.043478260869565216, 0.043478260869565216, 0.05857487922705313, 0.08269622449853348, 0.20222127405459506, 0.3367030600202325]
#         }

# plot(metrics)
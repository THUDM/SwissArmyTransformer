
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
if __name__ == "__main__":
    number_layer = 24
    # files = ["boolq","finetune_16000_boolq","bitfit_4000_boolq", "bitfit_16000_boolq", "pt_4000_boolq", "pt_16000_boolq"]
    files = ["diff_boolq_baseline"]
    names = []
    values = np.zeros([len(files), 12 * number_layer])
    for i, file in enumerate(files):
        with open(file+'.txt', "r") as f:
            lines = f.readlines()
        now_var = 0
        for line in lines:
            if line.startswith('.'):
                name, cnt, value = line[1:].split(' ')
                if line.find("query_key_value.bias") !=-1:
                    value = float(value) * 3 / 2
                values[i][now_var] = value
                now_var += 1
                name = name.replace('input_layernorm', "Inln")
                name = name.replace('query_key_value', "Qkv")
                name = name.replace('post_attention_layernorm', 'Postln')
                name = name.replace('dense_h_to_4h', "D1")
                name = name.replace('dense_4h_to_h', "D2")
                name = name.replace('weight', "W")
                name = name.replace('bias', "B")
                name = name.replace('attention', "At")
                if i == 0:
                    names.append(name)
    values = values.transpose()
    dir = "heatmap_diff"
    if not os.path.exists(dir):
        os.mkdir(dir)
    os.chdir(dir)

    for i in range(number_layer):
        plt.figure(figsize=(10,5))
        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)
        sns.heatmap(values[i*12:(i+1)*12], cmap='Reds', yticklabels =names[i*12 : (i+1)*12], xticklabels=files)
        plt.savefig(f'layer_{i}.jpg')
        plt.clf()

    plt.figure(figsize=(17,40))
    sns.heatmap(values, cmap='Reds', yticklabels =names, xticklabels=files)
    plt.savefig("heatmap.jpg")

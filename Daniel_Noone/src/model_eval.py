 # imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eval_model(model, loader, loss_fn, device):
    model.eval() # putting model in eval mode --> toff dout and batchnorm beh

    tot_loss = 0
    labels = []
    preds_all = []
    probs_all = []

    with torch.no_grad(): # not tracking grads to save mem
        for img, tab, y in loader: # looping thru val/tet batches
            img = img.to(device) 
            tab = tab.to(device)
            y = y.to(device) # sending to device.

            
            # fw pass
            logits = model(img, tab) # gtting logit output

            loss = loss_fn(logits, y) # comp loss

            probs = torch.softmax(logits, dim = 1) # logits > probs so can plot on hmap

            preds = torch.argmax(probs, dim = 1) # selecting class w highest prob

            # tracking loss - batcj
            tot_loss += loss.item()

            # tens back to cpu, storing for metrics
            labels.extend(y.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

        # mean loss -> over batches
        avg_l = tot_loss / len(loader)

        acc = accuracy_score(labels, preds_all) # calc std accuracy score

        # macro_f1 calc
        macro_f1 = f1_score(labels, preds_all, average = "macro", zero_division = 0)

        return avg_l, acc, macro_f1, labels, preds_all, probs_all

        
def plot_soft_prob_cf(true_labs, pred_probs, class_names, plot_title, color = "#ff3e01"):
    n_class = len(class_names) # num diseases

    soft_cmx = np.zeros((n_class, n_class)) # getting empty mx of zeros for conf mx

    true_labs = np.array(true_labs)
    pred_probs = np.array(pred_probs) # conv to np arrays

    # loping over each disease

    for i in range(n_class):
        dis_idx = np.where(true_labs == i)[0] # all px where actual label is class i 

        if len(dis_idx) > 0: 
            dis_prob = pred_probs[dis_idx] # getting prob vec or Px in true disease class i

            soft_cmx[i,:] = np.mean(dis_prob, axis = 0)

    cmap = sns.light_palette(color, as_cmap = True) # cusom cmap w specified color

    plt.figure(figsize = (8,8))

    sns.heatmap(soft_cmx, annot = True, fmt = ".1%", # show vlas in cell, and showing as pct (%)
               cmap = cmap, xticklabels = class_names, yticklabels = class_names) 

    plt.ylabel("True Disease")
    plt.xlabel("Average Predicted Probability")
    plt.title(plot_title)

    plt.savefig('avg_prob_hm.png', dpi = 300, bbox_inches = "tight")
    plt.show()



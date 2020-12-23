import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rand_bbox(size, lam):
    """
    Returns the coordinate values of a randomly drawn bounding box
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

class Mix:
    """
    Add CutMix and FMix here.
    """
    def cutmix(self, **args):
        pass
    def fmix(self, **args):
        pass

def plot_results(train_acc, valid_acc, train_loss, valid_loss, nb_epochs):
    epochs = [i for i in range(nb_epochs)]
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    
    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, valid_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    
    ax[1].plot(epochs, train_acc, 'go-', label='Training Loss')
    ax[1].plot(epochs, valid_acc, 'ro-', label='Validation Loss')
    ax[1].set_title('Training & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    
    plt.show()
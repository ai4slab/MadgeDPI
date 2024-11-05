import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


def plot_auc(y_true, y_score, epoch):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    au_prc = auc(recall, precision)

    plt.figure(figsize=(6, 6))
    plt.title('ROC Curve (GA-ENs, Epoch=%d)' % epoch, fontweight='bold', fontsize=16)
    plt.plot(fpr, tpr, '#037dfe', label='Test AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.xticks()
    plt.savefig('figs/ROC curve.png', dpi=600)
    # plt.show()
    plt.close()

    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.title('PR Curve (GA-ENs, Epoch=%d)' % epoch, fontweight='bold', fontsize=16)
    plt.plot(recall, precision, '#037dfe', label='Test AUPR = %0.4f' % au_prc)
    plt.legend(loc='lower left')
    plt.plot([0, 1], [1, 0], color='grey', linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Recall', fontsize=16)
    plt.xticks()
    plt.savefig('figs/PRC curve.png', dpi=600)
    # plt.show()
    plt.close()

    return roc_auc, au_prc

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

print("🧪 Loading data extracted during train_models.py...")
df = pd.read_pickle('Resume_Screening_Project/models/processed_resume_dataset.pkl')
y = df['Category']

print("🔪 Translating strings into Feature Vectors...")
tfidf_features = joblib.load('Resume_Screening_Project/models/dataset_tfidf_matrix.pkl')[1:, :]
bert_features = csr_matrix(joblib.load('Resume_Screening_Project/models/dataset_bert_embeddings.pkl'))
hybrid_features = hstack([tfidf_features, bert_features])

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    eps = 1e-9
    
    sens = np.mean(TP / (TP + FN + eps))
    spec = np.mean(TN / (TN + FP + eps))
    acc = accuracy_score(y_true, y_pred)
    prec = np.mean(TP / (TP + FP + eps))
    fpr = np.mean(FP / (FP + TN + eps))
    fnr = np.mean(FN / (FN + TP + eps))
    npv = np.mean(TN / (TN + FN + eps))
    fdr = np.mean(FP / (FP + TP + eps))
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'Sensitivity': sens, 'Specificity': spec, 'Accuracy': acc, 'Precision': prec,
        'FPR': fpr, 'FNR': fnr, 'NPV': npv, 'FDR': fdr, 'F1-Score': f1, 'MCC': mcc
    }

results = {'TF-IDF': {}, 'BERT': {}, 'Hybrid': {}}
train_pcts = [0.5, 0.6, 0.7, 0.8, 0.9]

for pct in train_pcts:
    print(f"⚙️ Training models on {int(pct*100)}% Data Split...")
    
    X_train_tf, X_test_tf, y_train, y_test = train_test_split(tfidf_features, y, train_size=pct, random_state=42, stratify=y)
    X_train_b, X_test_b, _, _ = train_test_split(bert_features, y, train_size=pct, random_state=42, stratify=y)
    X_train_h, X_test_h, _, _ = train_test_split(hybrid_features, y, train_size=pct, random_state=42, stratify=y)
    
    clf1 = LinearSVC(random_state=42, dual=True, max_iter=2000).fit(X_train_tf, y_train)
    results['TF-IDF'][pct] = compute_metrics(y_test, clf1.predict(X_test_tf))
    
    clf2 = LinearSVC(random_state=42, dual=True, max_iter=2000).fit(X_train_b, y_train)
    results['BERT'][pct] = compute_metrics(y_test, clf2.predict(X_test_b))
    
    clf3 = LinearSVC(random_state=42, dual=True, max_iter=2000).fit(X_train_h, y_train)
    results['Hybrid'][pct] = compute_metrics(y_test, clf3.predict(X_test_h))


def style_ax(ax, title, y_label):
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Training Data Split (%)', fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)

x_labels = ['50%', '60%', '70%', '80%', '90%']
x = np.arange(len(x_labels))
width = 0.5
color = '#3b82f6'

mets = {m: [results['Hybrid'][pct][m] for pct in train_pcts] for m in results['Hybrid'][0.5].keys()}

print("📊 Generating Authentic Plots...")
fig9, axs9 = plt.subplots(2, 2, figsize=(10, 8))
axs9[0,0].bar(x, mets['Sensitivity'], width, color=color); style_ax(axs9[0,0], 'Sensitivity (Recall)', ''); axs9[0,0].set_xticks(x, x_labels)
axs9[0,1].bar(x, mets['Specificity'], width, color=color); style_ax(axs9[0,1], 'Specificity', ''); axs9[0,1].set_xticks(x, x_labels)
axs9[1,0].bar(x, mets['Accuracy'], width, color=color); style_ax(axs9[1,0], 'Accuracy', ''); axs9[1,0].set_xticks(x, x_labels)
axs9[1,1].bar(x, mets['F1-Score'], width, color=color); style_ax(axs9[1,1], 'F1-Score', ''); axs9[1,1].set_xticks(x, x_labels)
plt.tight_layout(); fig9.savefig('fig9.png', dpi=300, bbox_inches='tight'); plt.close(fig9)

fig10, axs10 = plt.subplots(2, 2, figsize=(10, 8))
axs10[0,0].bar(x, mets['FDR'], width, color='#ef4444'); style_ax(axs10[0,0], 'FDR', ''); axs10[0,0].set_xticks(x, x_labels)
axs10[0,1].bar(x, mets['FNR'], width, color='#ef4444'); style_ax(axs10[0,1], 'FNR', ''); axs10[0,1].set_xticks(x, x_labels)
axs10[1,0].bar(x, mets['NPV'], width, color='#10b981'); style_ax(axs10[1,0], 'NPV', ''); axs10[1,0].set_xticks(x, x_labels)
axs10[1,1].bar(x, mets['FPR'], width, color='#ef4444'); style_ax(axs10[1,1], 'FPR', ''); axs10[1,1].set_xticks(x, x_labels)
plt.tight_layout(); fig10.savefig('fig10.png', dpi=300, bbox_inches='tight'); plt.close(fig10)

fig11, axs11 = plt.subplots(1, 2, figsize=(10, 4))
axs11[0].bar(x, mets['Precision'], width, color=color); style_ax(axs11[0], 'Precision', ''); axs11[0].set_xticks(x, x_labels)
axs11[1].bar(x, mets['MCC'], width, color=color); style_ax(axs11[1], 'MCC', ''); axs11[1].set_xticks(x, x_labels)
plt.tight_layout(); fig11.savefig('fig11.png', dpi=300, bbox_inches='tight'); plt.close(fig11)

print("\n--- TABLE 3 GENERATED MARKDOWN (For 80% Split) ---")
for metric in ['Specificity', 'Sensitivity', 'Accuracy', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']:
    tf = results['TF-IDF'][0.8][metric]
    bt = results['BERT'][0.8][metric]
    hy = results['Hybrid'][0.8][metric]
    print(f"| **{metric}** | {tf:.3f} | {bt:.3f} | {hy:.3f} |")
print("--------------------------------------------------")

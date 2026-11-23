import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from metrics import ece

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
base = pd.read_csv('results/predictions.csv'); aug = pd.read_csv('results/predictions_aug.csv')
LINES = [('Plain CNN', base, 'softmax', '#8A8F98', '-'), ('MC dropout', base, 'mc_dropout', '#E07B39', '-'),
         ('Deep ensemble', base, 'ensemble', '#2F6DB5', '-'), ('Deep ensemble + noise training', aug, 'ensemble', '#2F6DB5', '--')]
CONDS = ['clean', 20, 10, 5, 0, -5]; XL = ['clean', '20 dB', '10 dB', '5 dB', '0 dB', '-5 dB']


def at(d, c):
    return d[d.condition == 'clean'] if c == 'clean' else d[(d.condition == 'noisy') & (d.snr == c)]


fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
for name, df, m, col, ls in LINES:
    d = df[df.method == m]
    ax[0].plot(XL, [at(d, c).correct.mean() for c in CONDS], marker='o', color=col, ls=ls, label=name)
    ax[1].plot(XL, [ece(at(d, c).conf.values, at(d, c).correct.values) for c in CONDS], marker='o', color=col, ls=ls, label=name)
ax[0].set_ylim(0, 1); ax[0].set_ylabel('accuracy'); ax[0].set_title('Accuracy as household noise gets louder')
ax[1].set_ylabel('calibration error (lower is better)'); ax[1].set_title('Is the confidence honest?')
for a in ax: a.set_xlabel('test condition (signal-to-noise ratio)')
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('results/fig_noise.png', dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={'width_ratios': [1, 1.25]})
bars = []
for name, df, m, score in [('Plain CNN\n(confidence)', base, 'softmax', 'entropy'), ('MC dropout\n(disagreement)', base, 'mc_dropout', 'mutual_info'),
                           ('Deep ensemble\n(disagreement)', base, 'ensemble', 'mutual_info'), ('Ensemble + noise\ntraining', aug, 'ensemble', 'mutual_info')]:
    d = df[df.method == m]; cl, un = d[d.condition == 'clean'], d[d.condition == 'unseen']
    bars.append((name, roc_auc_score(np.r_[np.zeros(len(cl)), np.ones(len(un))], np.r_[cl[score], un[score]])))
ax[0].bar([b[0] for b in bars], [b[1] for b in bars], color=['#8A8F98', '#E07B39', '#2F6DB5', '#1F4E85'])
ax[0].axhline(0.5, color='#B03A2E', ls='--', lw=1); ax[0].text(3.45, 0.51, 'coin flip', fontsize=8, color='#B03A2E', ha='right')
for i, b in enumerate(bars): ax[0].text(i, b[1] + 0.01, f'{b[1]:.2f}', ha='center', fontsize=9)
ax[0].set_ylim(0.4, 0.85); ax[0].set_ylabel('AUROC'); ax[0].set_title('Spotting sounds it never learned'); ax[0].tick_params(axis='x', labelsize=7.5)
d = base[base.method == 'ensemble']; bins = np.linspace(0, d.mutual_info.max(), 30)
ax[1].hist(d[d.condition == 'clean'].mutual_info, bins=bins, alpha=0.6, color='#3A9D5D', density=True, label='known night sounds')
ax[1].hist(d[d.condition == 'unseen'].mutual_info, bins=bins, alpha=0.6, color='#B03A2E', density=True, label='sounds it never learned')
ax[1].set_xlabel('disagreement between the 5 models (mutual information)'); ax[1].set_ylabel('density')
ax[1].set_title('Deep ensemble: models disagree more on new sounds'); ax[1].legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig('results/fig_unseen.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 3.5))
for name, df, m, col, ls in [LINES[0], LINES[2], LINES[3]]:
    d = df[df.method == m].sort_values('entropy')
    bad = ((d.label == -1) | (d.correct == 0)).values.astype(float)
    cov = np.arange(1, len(d) + 1) / len(d)
    ax.plot(cov, np.cumsum(bad) / np.arange(1, len(d) + 1), color=col, ls=ls, label=name)
ax.axvline(0.8, color='#999', ls=':', lw=1); ax.text(0.79, 0.05, 'hand 20% to a human', fontsize=8, color='#555', ha='right')
ax.set_ylim(0, 0.75); ax.set_xlabel('share of clips the model decides on its own'); ax.set_ylabel('error rate on those clips')
ax.set_title('Realistic night mix: handing off the least certain clips'); ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig('results/fig_review.png', dpi=200); plt.close()
print('figures written')

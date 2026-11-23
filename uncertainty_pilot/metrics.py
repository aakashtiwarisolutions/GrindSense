import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

METHODS = ['softmax', 'mc_dropout', 'ensemble']


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def review_budget(df, frac, score='entropy'):
    order = df.sort_values(score, ascending=False)
    n = int(round(frac * len(order)))
    flagged, kept = order.iloc[:n], order.iloc[n:]
    bad = (order.label == -1) | (order.correct == 0)
    caught = ((flagged.label == -1) | (flagged.correct == 0)).sum()
    kept_known = kept[kept.label != -1]
    return dict(flag_fraction=frac, problems_total=int(bad.sum()), problems_caught=int(caught),
                catch_rate=float(caught / bad.sum()),
                kept_accuracy=float(kept_known.correct.mean()),
                kept_unseen=int((kept.label == -1).sum()))


def main(pred='results/predictions.csv', out_json='results/metrics.json'):
    df = pd.read_csv(pred)
    out = {'n_clips': {}, 'methods': {}}
    known = df[df.condition != 'unseen']
    for c in ['clean', 'noisy', 'unseen']:
        out['n_clips'][c] = int(df[(df.method == 'ensemble') & (df.condition == c)]['clip'].nunique() if c != 'noisy'
                                else len(df[(df.method == 'ensemble') & (df.condition == 'noisy')]))
    for m in METHODS:
        d = df[df.method == m]; r = {}
        cl = d[d.condition == 'clean']
        r['clean'] = dict(acc=cl.correct.mean(), ece=ece(cl.conf.values, cl.correct.values))
        for s in sorted(d.snr.dropna().unique(), reverse=True):
            z = d[(d.condition == 'noisy') & (d.snr == s)]
            r[f'snr_{int(s)}'] = dict(acc=z.correct.mean(), ece=ece(z.conf.values, z.correct.values),
                                     fail_auroc=roc_auc_score(1 - z.correct, z.entropy))
        un = d[d.condition == 'unseen']
        r['unseen_auroc_entropy'] = roc_auc_score(np.r_[np.zeros(len(cl)), np.ones(len(un))], np.r_[cl.entropy, un.entropy])
        if m != 'softmax':
            r['unseen_auroc_mi'] = roc_auc_score(np.r_[np.zeros(len(cl)), np.ones(len(un))], np.r_[cl.mutual_info, un.mutual_info])
        r['unseen_mean_conf'] = un.conf.mean()
        night = d.copy()
        r['review'] = [review_budget(night, f) for f in (0.1, 0.2, 0.3)]
        if m != 'softmax':
            r['review_mi'] = [review_budget(night, f, 'mutual_info') for f in (0.1, 0.2, 0.3)]
        r['night_no_review_accuracy_on_known'] = night[night.label != -1].correct.mean()
        out['methods'][m] = r
    per_fold = df[(df.condition == 'clean')].groupby(['method', 'fold']).correct.mean().unstack()
    out['clean_acc_per_fold'] = per_fold.round(3).to_dict()
    json.dump(out, open(out_json, 'w'), indent=2, default=float)
    print(json.dumps(out, indent=1, default=lambda v: round(float(v), 3)))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import features as fx
from model import GrindCNN, mc_dropout_mode

torch.set_num_threads(2)
SNRS = [20, 10, 5, 0, -5]
MEMBERS, MC_PASSES = 5, 30
MODELS = sys.argv[1] if len(sys.argv) > 1 else 'models'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results/predictions.csv'
K = len(fx.IN_DIST)


def mix(x, n, snr):
    px, pn = np.mean(x ** 2), np.mean(n ** 2) + 1e-12
    return x + n * np.sqrt(px / (pn * 10 ** (snr / 10)))


def load_members(fold):
    out = []
    for m in range(MEMBERS):
        ck = torch.load(f'{MODELS}/fold{fold}_m{m}.pt')
        net = GrindCNN(K); net.load_state_dict(ck['state']); net.eval()
        out.append((net, ck['mu'], ck['sd']))
    return out


def batch_probs(net, X, passes=1, mc=False):
    with torch.no_grad():
        xt = torch.tensor(X)[:, None]
        if mc:
            mc_dropout_mode(net)
            p = torch.stack([F.softmax(net(xt), -1) for _ in range(passes)])
            net.eval()
            return p.numpy()
        return F.softmax(net(xt), -1).numpy()[None]


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(-1)


def summarise(samples):
    mean = samples.mean(0)
    pe = entropy(mean)
    mi = pe - entropy(samples).mean(0)
    return mean, pe, mi


def main():
    clips = pd.read_csv('data/clips.csv'); waves = np.load('data/waves.npy')
    noise_idx = np.where(clips.category.isin(fx.NOISE))[0]
    rows = []
    for fold in range(1, 6):
        members = load_members(fold)
        mu, sd = members[0][1], members[0][2]
        test_in = np.where(clips.category.isin(fx.IN_DIST) & (clips.fold == fold))[0]
        test_un = np.where(clips.category.isin(fx.UNSEEN) & (clips.fold == fold))[0]
        rng = np.random.default_rng(fold)
        assign = rng.choice(noise_idx, len(test_in))
        sets = [('clean', None, test_in, [waves[i] for i in test_in])]
        for s in SNRS:
            sets.append(('noisy', s, test_in, [mix(waves[i], waves[n], s) for i, n in zip(test_in, assign)]))
        sets.append(('unseen', None, test_un, [waves[i] for i in test_un]))
        for cond, snr, idx, ws in sets:
            X = ((np.stack([fx.logmel(w) for w in ws]) - mu) / sd).astype(np.float32)
            y = np.array([fx.IN_DIST.index(c) if c in fx.IN_DIST else -1 for c in clips.category.values[idx]])
            outs = {
                'softmax': batch_probs(members[0][0], X),
                'mc_dropout': batch_probs(members[0][0], X, MC_PASSES, mc=True),
                'ensemble': np.concatenate([batch_probs(m[0], X) for m in members]),
            }
            for method, samp in outs.items():
                mean, pe, mi = summarise(samp)
                for j in range(len(idx)):
                    rows.append(dict(fold=fold, method=method, condition=cond, snr=snr if snr is not None else np.nan,
                                     clip=int(idx[j]), category=clips.category.values[idx[j]], label=int(y[j]),
                                     pred=int(mean[j].argmax()), conf=float(mean[j].max()),
                                     entropy=float(pe[j]), mutual_info=float(mi[j])))
        print('fold', fold, 'evaluated', flush=True)
    df = pd.DataFrame(rows)
    df['correct'] = (df.pred == df.label).astype(int)
    df.to_csv(OUT, index=False)


if __name__ == '__main__':
    main()

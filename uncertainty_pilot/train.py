import sys, json, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
import features as fx
from model import GrindCNN

torch.set_num_threads(2)
CROP, EPOCHS, BATCH = 150, 60, 32


def spec_augment(x, rng):
    x = x.copy()
    for _ in range(2):
        t = rng.integers(0, 20); t0 = rng.integers(0, x.shape[1] - t + 1); x[:, t0:t0 + t] = 0
        f = rng.integers(0, 8); f0 = rng.integers(0, x.shape[0] - f + 1); x[f0:f0 + f, :] = 0
    return x


def train_member(X, y, seed, n_classes):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = GrindCNN(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // BATCH)):
            xb = []
            for i in b:
                s = rng.integers(0, X[i].shape[1] - CROP + 1)
                xb.append(spec_augment(X[i][:, s:s + CROP] + rng.normal(0, 0.3), rng))
            xb = torch.tensor(np.stack(xb))[:, None]
            loss = F.cross_entropy(model(xb), torch.tensor(y[b]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def main(fold, members):
    clips = pd.read_csv('data/clips.csv'); waves = np.load('data/waves.npy')
    ind = clips.category.isin(fx.IN_DIST).values
    tr = np.where(ind & (clips.fold != fold).values)[0]
    mels = np.stack([fx.logmel(waves[i]) for i in tr])
    mu, sd = float(mels.mean()), float(mels.std())
    X = (mels - mu) / sd
    y = np.array([fx.IN_DIST.index(c) for c in clips.category.values[tr]])
    for m in range(members):
        t = time.time()
        model = train_member(X, y, seed=1000 * fold + m, n_classes=len(fx.IN_DIST))
        torch.save({'state': model.state_dict(), 'mu': mu, 'sd': sd}, f'models/fold{fold}_m{m}.pt')
        print(f'fold {fold} member {m} done in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))

import sys, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
import features as fx
from model import GrindCNN
from train import spec_augment, EPOCHS, BATCH

torch.set_num_threads(2)
TRAIN_NOISE = ['sea_waves', 'crackling_fire', 'water_drops', 'pouring_water', 'crickets', 'chirping_birds']
CROP_S = 48000


def load_train_noise(root):
    meta = pd.read_csv(Path(root) / 'meta' / 'esc50.csv')
    files = meta[meta.category.isin(TRAIN_NOISE)].filename
    return np.stack([fx.load(Path(root) / 'audio' / f) for f in files])


def main(fold, members, root):
    clips = pd.read_csv('data/clips.csv'); waves = np.load('data/waves.npy')
    noise = load_train_noise(root)
    ind = clips.category.isin(fx.IN_DIST).values
    tr = np.where(ind & (clips.fold != fold).values)[0]
    mels = np.stack([fx.logmel(waves[i]) for i in tr])
    mu, sd = float(mels.mean()), float(mels.std())
    y_all = np.array([fx.IN_DIST.index(c) for c in clips.category.values[tr]])
    W = waves[tr]
    for m in range(members):
        t = time.time(); seed = 5000 + 1000 * fold + m
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        model = GrindCNN(len(fx.IN_DIST))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        for ep in range(EPOCHS):
            model.train()
            for b in np.array_split(rng.permutation(len(W)), max(1, len(W) // BATCH)):
                xb = []
                for i in b:
                    s = rng.integers(0, W.shape[1] - CROP_S + 1); x = W[i, s:s + CROP_S]
                    if rng.random() < 0.5:
                        n = noise[rng.integers(len(noise))]; ns = rng.integers(0, n.shape[0] - CROP_S + 1)
                        from evaluate import mix
                        x = mix(x, n[ns:ns + CROP_S], rng.uniform(0, 20))
                    xb.append(spec_augment((fx.logmel(x)[:, :150] - mu) / sd, rng))
                xb = torch.tensor(np.stack(xb).astype(np.float32))[:, None]
                loss = F.cross_entropy(model(xb), torch.tensor(y_all[b]))
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        torch.save({'state': model.state_dict(), 'mu': mu, 'sd': sd}, f'models_aug/fold{fold}_m{m}.pt')
        print(f'aug fold {fold} member {m} done in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

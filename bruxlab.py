import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter

SR = 16000
FRAME = 800
FPS = SR // FRAME
CLIP_S = 3.0
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
AUDIO_EXT = {'.m4a', '.wav', '.mp3', '.aac', '.caf', '.mp4', '.flac', '.ogg'}
LABELS = {'g': 'grinding', 'o': 'other', 'u': 'unsure'}


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def stream(path, chunk_s=60):
    cmd = [ffmpeg(), '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SR), '-f', 's16le', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = SR * chunk_s * 2
    while True:
        b = p.stdout.read(size)
        if not b:
            break
        yield np.frombuffer(b[: len(b) // 2 * 2], dtype=np.int16)
    p.wait()
    if p.returncode != 0:
        sys.exit(f'Could not read {path}. Is it an audio file?')


def frame_db(x):
    n = len(x) // FRAME
    f = x[: n * FRAME].astype(np.float32).reshape(n, FRAME) / 32768.0
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-6)


def noise_floor(db):
    n = len(db) // FPS
    if n < 3:
        return np.full_like(db, np.percentile(db, 20))
    blocks = np.percentile(db[: n * FPS].reshape(n, FPS), 20, axis=1)
    smooth = median_filter(blocks, size=min(61, n if n % 2 else n - 1), mode='nearest')
    floor = np.repeat(smooth, FPS)
    return np.concatenate([floor, np.full(len(db) - len(floor), smooth[-1])])


def find_events(db, floor, over_db, min_db):
    on = (db > floor + over_db) & (db > min_db)
    events, start, gap = [], None, 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap = 0
            end = i
        elif start is not None:
            gap += 1
            if gap > FPS // 2:
                events.append((start, end))
                start = None
    if start is not None:
        events.append((start, end))
    return [(s, e) for s, e in events if e - s + 1 >= FPS // 4]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def cut_clips(path, windows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_len = int(CLIP_S * SR)
    order = sorted(range(len(windows)), key=lambda k: windows[k][0])
    buf = np.zeros(0, dtype=np.int16)
    buf_start = 0
    k = 0
    for chunk in stream(path):
        buf = np.concatenate([buf, chunk])
        while k < len(order):
            name, s = windows[order[k]][1], int(windows[order[k]][0] * SR)
            if s + clip_len > buf_start + len(buf):
                break
            a = max(0, s - buf_start)
            wavfile.write(out_dir / f'{name}.wav', SR, buf[a: a + clip_len])
            k += 1
        keep = clip_len + SR
        if len(buf) > keep:
            drop = len(buf) - keep
            if k < len(order):
                drop = min(drop, max(0, int(windows[order[k]][0] * SR) - buf_start))
            buf = buf[drop:]
            buf_start += drop
    while k < len(order):
        name, s = windows[order[k]][1], int(windows[order[k]][0] * SR)
        a = max(0, s - buf_start)
        seg = buf[a: a + clip_len]
        if len(seg) > SR // 2:
            wavfile.write(out_dir / f'{name}.wav', SR, np.pad(seg, (0, clip_len - len(seg))))
        k += 1


def prepare(args):
    path = Path(args.recording)
    night = args.night or path.stem
    print(f'Reading {path.name} ...')
    t0 = time.time()
    db = np.concatenate([frame_db(c) for c in stream(path)])
    hours = len(db) / FPS / 3600
    floor = noise_floor(db)
    events = find_events(db, floor, args.over_db, args.min_db)
    cands = []
    for i, (s, e) in enumerate(events):
        peak = float((db[s: e + 1] - floor[s: e + 1]).max())
        dur = (e - s + 1) / FPS
        start = s / FPS
        n = min(args.max_per_event, max(1, int(np.ceil(dur / CLIP_S))))
        for j in range(n):
            cs = max(0.0, start - 0.5 + j * CLIP_S)
            cands.append({'start_s': cs, 'event_id': i, 'event_dur_s': round(dur, 2), 'peak_over_floor_db': round(peak, 1), 'kind': 'event'})
    if len(cands) > args.max_clips:
        print(f'{len(cands)} candidates found; keeping the {args.max_clips} loudest. Raise --max-clips to keep more.')
        cands = sorted(cands, key=lambda c: -c['peak_over_floor_db'])[: args.max_clips]
    busy = np.zeros(len(db), bool)
    for s, e in events:
        busy[max(0, s - 2 * FPS): e + 2 * FPS] = True
    quiet = np.where(~busy[: max(0, len(db) - int(CLIP_S * FPS))])[0]
    rng = np.random.default_rng(0)
    for s in rng.choice(quiet, size=min(args.quiet, len(quiet)), replace=False) if len(quiet) else []:
        cands.append({'start_s': s / FPS, 'event_id': -1, 'event_dur_s': 0, 'peak_over_floor_db': 0, 'kind': 'quiet'})
    cands.sort(key=lambda c: c['start_s'])
    for i, c in enumerate(cands):
        c['clip'] = f'{night}_{i:04d}'
        c['night'] = night
        c['start_s'] = round(c['start_s'], 2)
        c['time'] = time.strftime('%H:%M:%S', time.gmtime(c['start_s']))
    cut_clips(path, [(c['start_s'], c['clip']) for c in cands], DATA / 'clips' / night)
    fields = ['clip', 'night', 'start_s', 'time', 'kind', 'event_id', 'event_dur_s', 'peak_over_floor_db']
    write_csv(DATA / 'clips' / night / 'clips.csv', cands, fields)
    np.save(DATA / 'clips' / night / 'loudness_db.npy', db.astype(np.float16))
    n_ev = sum(c['kind'] == 'event' for c in cands)
    print(f'Night {night}: {hours:.1f} h of audio, {len(events)} sound events, {n_ev} event clips + {len(cands) - n_ev} quiet clips ({time.time() - t0:.0f} s).')
    print(f'Next: python bruxlab.py label {night}')


def calibrate(args):
    folder = Path(args.folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f'No audio files in {folder}')
    rows = []
    out = DATA / 'clips' / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    for p in files:
        sound = p.stem.lower().split('_')[0].split(' ')[0]
        label = 'grinding' if sound.startswith('grind') else 'other'
        x = np.concatenate(list(stream(p)))
        db = frame_db(x)
        floor = np.percentile(db, 20)
        clip_len, hop = int(CLIP_S * SR), int(CLIP_S * SR / 2)
        n = 0
        for s in range(0, max(1, len(x) - clip_len + 1), hop):
            seg = x[s: s + clip_len]
            seg_db = frame_db(seg)
            if sound != 'silence' and seg_db.max() < floor + args.over_db:
                continue
            name = f'calib_{sound}_{n:03d}'
            wavfile.write(out / f'{name}.wav', SR, np.pad(seg, (0, clip_len - len(seg))))
            rows.append({'clip': name, 'night': 'calibration', 'label': label, 'sound': sound, 'labeled_at': 'calibration'})
            n += 1
        print(f'{p.name}: {n} clips labeled "{label}" ({sound})')
    write_csv(DATA / 'labels' / 'calibration.csv', rows, ['clip', 'night', 'label', 'sound', 'labeled_at'])
    print(f'Saved {len(rows)} calibration clips.')


def play(path):
    if sys.platform == 'win32':
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif sys.platform == 'darwin':
        subprocess.Popen(['afplay', str(path)])
    else:
        for player in (['paplay'], ['aplay', '-q']):
            try:
                subprocess.Popen(player + [str(path)], stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue


def stop():
    if sys.platform == 'win32':
        import winsound
        winsound.PlaySound(None, 0)


def label(args):
    import tkinter as tk
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    night = args.night
    clip_dir = DATA / 'clips' / night
    clips = read_csv(clip_dir / 'clips.csv')
    if not clips:
        sys.exit(f'No clips for {night}. Run: python bruxlab.py prepare <recording>')
    lab_path = DATA / 'labels' / f'{night}.csv'
    labels = {r['clip']: r for r in read_csv(lab_path)}
    order = [c for c in clips if c['kind'] == 'event'] + [c for c in clips if c['kind'] == 'quiet']
    if args.loudest_first:
        order.sort(key=lambda c: -float(c['peak_over_floor_db']))
    state = {'i': next((k for k, c in enumerate(order) if c['clip'] not in labels), len(order))}

    def save():
        rows = [labels[c['clip']] for c in clips if c['clip'] in labels]
        write_csv(lab_path, rows, ['clip', 'night', 'label', 'sound', 'labeled_at'])

    root = tk.Tk()
    root.title(f'GrindSense labeling: {night}')
    info = tk.Label(root, font=('Segoe UI', 12), justify='left', anchor='w')
    info.pack(fill='x', padx=10, pady=(10, 0))
    fig = Figure(figsize=(8, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill='both', expand=True, padx=10)
    help_text = 'G = grinding    O = other sound    U = unsure    R = replay    B = back    Q = save and quit'
    tk.Label(root, text=help_text, font=('Segoe UI', 11)).pack(pady=8)

    def show():
        i = state['i']
        done = len([c for c in order if c['clip'] in labels])
        g = sum(1 for r in labels.values() if r['label'] == 'grinding')
        if i >= len(order):
            info.config(text=f'All {len(order)} clips labeled ({g} grinding). Press Q to save and quit.')
            ax.clear()
            canvas.draw()
            return
        c = order[i]
        path = clip_dir / f"{c['clip']}.wav"
        sr, x = wavfile.read(path)
        current = labels.get(c['clip'], {}).get('label', '-')
        info.config(text=f"Clip {i + 1} of {len(order)}   |   time in recording {c['time']}   |   {c['kind']}, {c['peak_over_floor_db']} dB above background\n"
                         f"Labeled so far: {done}   (grinding: {g})   |   this clip: {current}")
        ax.clear()
        ax.specgram(x.astype(np.float32) / 32768.0, Fs=sr, NFFT=512, noverlap=384, cmap='magma')
        ax.set_ylim(0, 8000)
        ax.set_xlabel('seconds')
        ax.set_ylabel('Hz')
        fig.tight_layout()
        canvas.draw()
        play(path)

    def mark(lab):
        if state['i'] >= len(order):
            return
        c = order[state['i']]
        labels[c['clip']] = {'clip': c['clip'], 'night': night, 'label': lab, 'sound': '', 'labeled_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        save()
        state['i'] += 1
        show()

    def key(e):
        k = e.keysym.lower()
        if k in LABELS:
            mark(LABELS[k])
        elif k == 'r':
            show()
        elif k == 'b':
            state['i'] = max(0, state['i'] - 1)
            show()
        elif k == 'q':
            stop()
            save()
            root.destroy()

    root.bind('<Key>', key)
    root.protocol('WM_DELETE_WINDOW', lambda: (stop(), save(), root.destroy()))
    show()
    root.mainloop()
    print(f'Saved {len(labels)} labels to {lab_path}')


def stats(args):
    rows = []
    for p in sorted((DATA / 'labels').glob('*.csv')):
        r = read_csv(p)
        total = len(read_csv(DATA / 'clips' / p.stem / 'clips.csv')) if p.stem != 'calibration' else len(r)
        g = sum(x['label'] == 'grinding' for x in r)
        rows.append((p.stem, len(r), total, g, sum(x['label'] == 'other' for x in r), sum(x['label'] == 'unsure' for x in r)))
    if not rows:
        print('No labels yet.')
        return
    print(f"{'night':<14}{'labeled':>9}{'of':>6}{'grinding':>10}{'other':>8}{'unsure':>8}")
    for n, l, t, g, o, u in rows:
        print(f'{n:<14}{l:>9}{t:>6}{g:>10}{o:>8}{u:>8}')


def main():
    ap = argparse.ArgumentParser(description='Prepare and label overnight recordings for teeth-grinding detection.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare', help='find sound events in one night recording and cut them into short clips')
    p.add_argument('recording')
    p.add_argument('--night', help='name for this night (default: file name)')
    p.add_argument('--over-db', type=float, default=8.0, help='how many dB above background counts as an event')
    p.add_argument('--min-db', type=float, default=-65.0, help='ignore anything quieter than this (dBFS)')
    p.add_argument('--max-per-event', type=int, default=4)
    p.add_argument('--max-clips', type=int, default=600)
    p.add_argument('--quiet', type=int, default=40, help='number of random quiet clips to add')
    p.set_defaults(func=prepare)
    p = sub.add_parser('calibrate', help='cut the awake calibration recordings into labeled clips')
    p.add_argument('folder')
    p.add_argument('--over-db', type=float, default=6.0)
    p.set_defaults(func=calibrate)
    p = sub.add_parser('label', help='listen to clips and label them')
    p.add_argument('night')
    p.add_argument('--loudest-first', action='store_true')
    p.set_defaults(func=label)
    p = sub.add_parser('stats', help='show how many clips are labeled')
    p.set_defaults(func=stats)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

# GrindSense
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24288332.svg)](https://doi.org/10.5281/zenodo.24288332)

Private, on-device detection of sleep bruxism from smartphone audio.

This repository contains the data collection and labeling tools for GrindSense. They are used to record nights at home with an iPhone and label teeth-grinding sounds, building the dataset for a compact CNN that will run fully offline on the phone. The full research proposal is in `GrindSense_Research_Plan.pdf`.

Recordings and labels stay on this computer. The `data` folder is excluded from Git.

## Setup (Windows)

```
pip install -r requirements.txt
```

## Folder layout

```
bruxlab.py
data/
  raw/            night recordings from the iPhone (one file per night)
  calibration/    short awake recordings, one file per sound
  clips/          3-second clips cut from each night (created by the tool)
  labels/         your labels, one CSV per night (created by the tool)
```

## 1. Calibration (once)

Record each sound below as a separate Voice Memo of about 30 seconds while awake, and rename each memo to the sound name:

| File name | What to record |
|---|---|
| `grinding` | gentle side-to-side grinding |
| `clenching` | clenching and tapping teeth |
| `swallowing` | swallowing, coughing, lip smacking |
| `snoring` | snoring sounds |
| `movement` | turning over, sheets rustling |
| `silence` | the quiet room at night |

Put the files in `data/calibration/`, then run:

```
python bruxlab.py calibrate data/calibration
```

Files whose name starts with "grind" are labeled grinding; everything else is labeled other.

## 2. Each night

1. iPhone: Settings, Apps, Voice Memos, Audio Quality: Compressed.
2. Start a Voice Memo before sleep. Phone on the nightstand on her side, 30 to 50 cm from her head, screen down, plugged in, Do Not Disturb on.
3. In the morning, stop it and rename it with the date, for example `2026-11-17`.
4. Voice Memos, Share, Save to Files, iCloud Drive. Download it on the laptop from iCloud Drive into `data/raw/`.
5. Write the morning note in `data/diary.csv`: date, jaw pain (1-5), stress the evening before (1-5), caffeine or alcohol after 6 PM (yes/no).

## 3. Prepare and label a night

```
python bruxlab.py prepare data/raw/2026-11-17.m4a
python bruxlab.py label 2026-11-17
```

`prepare` finds every sound that stands out from the background, cuts it into 3-second clips, and adds 40 random quiet clips. An 8-hour night takes about a minute.

`label` opens a window that plays each clip and shows its spectrogram:

| Key | Meaning |
|---|---|
| G | grinding |
| O | other sound |
| U | unsure |
| R | replay |
| B | back to the previous clip |
| Q | save and quit |

Labels are saved after every key press, so you can stop and continue later. Grinding usually looks like repeated vertical bands between 1,000 and 6,000 Hz; snoring sits low, under 1,000 Hz.

Useful options:

```
python bruxlab.py label 2026-11-17 --loudest-first
python bruxlab.py prepare data/raw/2026-11-17.m4a --over-db 10
python bruxlab.py stats
```

Raise `--over-db` if a night gives too many clips (for example a noisy fan), lower it if quiet grinding is missed.

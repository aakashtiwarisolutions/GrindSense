# GrindSense

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24288332.svg)](https://doi.org/10.5281/zenodo.24288332)

Private, on-device detection of sleep bruxism from smartphone audio.

Many people grind their teeth at night without knowing it, and the standard way to check is a night in a sleep lab. GrindSense asks whether an ordinary iPhone on the nightstand can hear teeth grinding reliably, using a small convolutional neural network that runs fully offline, so no bedroom audio ever leaves the phone.

## What is in this repository

| Part | What it is |
|---|---|
| [GrindSense_Research_Plan.pdf](GrindSense_Research_Plan.pdf) | The full research plan: questions, methods, evaluation, ethics and timeline |
| `bruxlab.py` | My recording and labeling tool: finds sound events in a night recording, cuts them into 3-second clips and lets me label each one |
| [uncertainty_pilot/](uncertainty_pilot/) | A pilot study: does the GrindSense CNN know when it is confused under household noise and unfamiliar sounds? |

## Status

- Recording and labeling tool: done
- Uncertainty pilot on public night sounds: done ([results](uncertainty_pilot/))
- Recording real nights: in progress
- Training the CNN on real nights: next
- Offline iPhone app with Core ML: after that

## Pilot study

Before testing on real nights, I ran the GrindSense CNN on eight public night sounds and tested it under household noise it had never heard and on sounds it had never learned. Accuracy fell from 78% in a quiet room to 33% when the noise was as loud as the sound. A single model could not tell when it heard something new (AUROC 0.52), but disagreement inside a small ensemble could (0.73). The full write-up and code are in [uncertainty_pilot](uncertainty_pilot/).

## Privacy

Recordings and labels stay on my own computer. The `data` folder and all audio files are excluded from Git, and no audio is ever uploaded. The final app is designed to keep only event times and scores, and delete the audio within seconds.

## Setup (Windows)

~~~
pip install -r requirements.txt
~~~

## Folder layout

~~~
bruxlab.py
data/
  raw/            night recordings from the iPhone (one file per night)
  calibration/    short awake recordings, one file per sound
  clips/          3-second clips cut from each night (created by the tool)
  labels/         labels, one CSV per night (created by the tool)
uncertainty_pilot/
~~~

## 1. Calibration (once)

Record each sound below as a separate Voice Memo of about 30 seconds while awake, and name each memo after the sound:

| File name | What to record |
|---|---|
| `grinding` | gentle side-to-side grinding |
| `clenching` | clenching and tapping teeth |
| `swallowing` | swallowing, coughing, lip smacking |
| `snoring` | snoring sounds |
| `movement` | turning over, sheets rustling |
| `silence` | the quiet room at night |

Put the files in `data/calibration/`, then run:

~~~
python bruxlab.py calibrate data/calibration
~~~

Files whose name starts with "grind" are labeled grinding; everything else is labeled other.

## 2. Each night

1. iPhone: Settings, Apps, Voice Memos, Audio Quality: Compressed.
2. Start a Voice Memo before sleep. Phone on the nightstand, 30 to 50 cm from the sleeper's head, screen down, plugged in, Do Not Disturb on.
3. In the morning, stop it and rename it with the date, for example `2026-11-17`.
4. Voice Memos, Share, Save to Files, iCloud Drive. Download it on the laptop into `data/raw/`.
5. Add a morning note to `data/diary.csv`: date, jaw pain (1-5), stress the evening before (1-5), caffeine or alcohol after 6 PM (yes/no).

## 3. Prepare and label a night

~~~
python bruxlab.py prepare data/raw/2026-11-17.m4a
python bruxlab.py label 2026-11-17
~~~

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

~~~
python bruxlab.py label 2026-11-17 --loudest-first
python bruxlab.py prepare data/raw/2026-11-17.m4a --over-db 10
python bruxlab.py stats
~~~

Raise `--over-db` if a night gives too many clips (for example a noisy fan), and lower it if quiet grinding is missed.

## Author

Aakash Tiwari, M.S. Cybersecurity, Rowan University
[LinkedIn](https://www.linkedin.com/in/aakash-tiwari-46b6b916b/) · [GitHub](https://github.com/aakashtiwarisolutions)

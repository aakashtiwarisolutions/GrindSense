# Does my sleep-sound CNN know when it is confused?

A small pilot I ran before testing GrindSense on real nights. A bedroom is noisy, and a model that hears something new will still give an answer, often a confident one. I wanted to know how much noise hurts the GrindSense CNN, and whether it can tell me when not to trust it.

Write-up: [GrindSense_Uncertainty_Pilot.pdf](GrindSense_Uncertainty_Pilot.pdf)

Documentation (how it works, how to run it, every file explained): [GrindSense_Uncertainty_Pilot_Documentation.pdf](GrindSense_Uncertainty_Pilot_Documentation.pdf)

## What I did

There is no public teeth-grinding audio, so I used eight everyday night sounds from ESC-50 as a stand-in (snoring, breathing, coughing, sneezing, clock tick, door creak, footsteps, keyboard typing) and trained the same CNN I planned for GrindSense on log-mel spectrograms.

I compared a plain CNN, MC dropout and a deep ensemble of five CNNs, using 5-fold cross-validation. Then I tested them on clean clips, on the same clips mixed with household noise they never heard (rain, wind, washing machine, vacuum, engine, thunder), and on sounds they never learned (dog, cat, crying baby, alarm, siren, car horn, laughter, toilet flush). In a second round I trained with a different set of background noises to see if that helps.

## What I found

| | Plain CNN | Deep ensemble | Ensemble + noise training |
|---|---|---|---|
| Accuracy, clean | 77.5% | 78.1% | 76.9% |
| Accuracy, noise as loud as the sound (0 dB) | 33.1% | 32.5% | 37.5% |
| Calibration error at 0 dB (lower is better) | 0.41 | 0.37 | 0.18 |
| Spotting unseen sounds (AUROC, 0.5 is a coin flip) | 0.52 | 0.73 | 0.76 |
| Problems caught when 20% of clips go to a person | 24.4% | 25.7% | 27.1% |

Noise hurts a lot. The plain CNN's confidence cannot tell when it is hearing something new, but disagreement inside an ensemble can. Training with background noise makes the confidence much more honest without adding much accuracy. And handing uncertain clips to a person only helps a little, so the real test still has to happen on real nights.

## Running it

You need the ESC-50 audio (`git clone https://github.com/karolpiczak/ESC-50.git`) and `pip install numpy scipy pandas scikit-learn matplotlib torch`.

~~~
mkdir -p data models models_aug results
python features.py ESC-50 data
for f in 1 2 3 4 5; do python train.py $f 5; done
for f in 1 2 3 4 5; do python train_noiseaug.py $f 5 ESC-50; done
python evaluate.py models results/predictions.csv
python evaluate.py models_aug results/predictions_aug.csv
python metrics.py results/predictions.csv results/metrics.json
python metrics.py results/predictions_aug.csv results/metrics_aug.json
python figures.py
~~~

Each round of training took about an hour on a 2-core CPU. The ESC-50 audio is not included here because it has its own license (CC BY-NC 3.0).

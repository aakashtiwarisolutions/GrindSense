import torch, torch.nn as nn


class GrindCNN(nn.Module):
    def __init__(self, n_classes, p_block=0.2, p_head=0.5):
        super().__init__()
        layers, c_in = [], 1
        for c_out in (32, 64, 128, 128):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                       nn.MaxPool2d(2), nn.Dropout(p_block)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_head), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def mc_dropout_mode(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

"""Completed starter model and training helpers, plus our hard top-k variant.

Course TODO completion and project extensions were developed with Codex assistance.
The class interfaces come from the course notebook. E1/E3 class weights are
set in experiments/train.py, so all four variants share this model code.
"""
import math
import torch
from torch import nn
from utils import classification_metrics


class Classifier1FC(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.):
        super().__init__()
        self.dropout = nn.Dropout(droprate)
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        # apply dropout before the final slide classifier
        features = self.dropout(x)
        logits = self.fc(features)

        return logits


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim):
        super().__init__()
        self.fc = nn.Linear(n_channels, m_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # project each patch, then apply the nonlinearity
        projected = self.fc(x)
        features = self.relu(projected)

        return features


class AttentionGated(nn.Module):
    # gated attention from Ilse et al., ICML 2018, following the course scaffold
    # https://proceedings.mlr.press/v80/ilse18a.html
    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):
        # get one attention score for each patch
        tanh_features = self.attention_V(x)
        sigmoid_features = self.attention_U(x)

        gated = tanh_features * sigmoid_features
        return self.attention_weights_w(gated).transpose(1, 2)


class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)
        self.top_fraction = float(getattr(conf, 'top_fraction', 1.))
        if not 0 < self.top_fraction <= 1:
            raise ValueError('top_fraction must be in (0,1]')

    def forward(self, x, return_attention=False, mask=None):
        """Return slide logits, and normalized patch weights when requested."""
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if not mask.any(1).all():
            raise ValueError('Empty bag')
        h = self.dimreduction(x)
        # padding must not compete with real patches in softmax
        a = self.attention(h).masked_fill(~mask[:, None, :], float('-inf'))
        if self.top_fraction < 1:
            # our experimental hard top-k mask, not sparsemax or a paper implementation
            # keep ceil(r * valid patches), then normalize only the retained scores
            keep = torch.zeros_like(mask)
            for b in range(x.shape[0]):
                k = max(1, math.ceil(int(mask[b].sum()) * self.top_fraction))
                keep[b, a[b, 0].topk(k).indices] = True
            a = a.masked_fill(~keep[:, None, :], float('-inf'))
        # normalize over the retained patches
        attention = a.softmax(-1)

        # weighted sum gives one representation per slide
        pooled_features = torch.bmm(attention, h)
        bag = pooled_features.squeeze(1)

        logits = self.classifier(bag)
        return (logits, attention) if return_attention else logits


def train_one_epoch(model, criterion, data_loader, optimizer, device):
    """Run one optimization epoch; return average loss and accuracy in percent."""
    model.train()
    loss_sum = correct = n = 0
    max_mps_driver = max_mps_tensor = 0
    for data in data_loader:
        x = data['input'].to(device)
        y = data['label'].to(device)

        # clear the previous bag's gradients
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite input features')
        # padding is ignored by the attention module
        valid_patches = data['mask'].to(device)
        logits = model(x, mask=valid_patches)

        loss = criterion(logits, y)
        # mean over bags explicitly: weighted CE's default mean cancels weights at B=1
        if loss.ndim:
            loss = loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if device.type == 'mps':
            max_mps_driver = max(max_mps_driver, torch.mps.driver_allocated_memory())
            max_mps_tensor = max(max_mps_tensor, torch.mps.current_allocated_memory())
        optimizer.step()
        n += len(y)
        loss_sum += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
    return {'loss': loss_sum / n, 'acc': 100 * correct / n,
            'mps_driver_bytes_observed': max_mps_driver,
            'mps_tensor_bytes_observed': max_mps_tensor}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    """Return the course metrics and average loss without updating weights."""
    model.eval()
    probs, labels = [], []
    total = n = 0
    for data in data_loader:
        y = data['label'].to(device)
        # evaluation uses the same inputs and padding mask as training
        features = data['input'].to(device)
        valid_patches = data['mask'].to(device)
        logits = model(features, mask=valid_patches)
        total += criterion(logits, y).item() * len(y)
        n += len(y)
        # probabilities are normalized over classes, not patches
        probabilities = logits.softmax(1)
        probs.append(probabilities.cpu())
        labels.append(y.cpu())
    metrics = classification_metrics(torch.cat(probs), torch.cat(labels), conf.n_class)
    metrics['loss'] = total / n
    return metrics

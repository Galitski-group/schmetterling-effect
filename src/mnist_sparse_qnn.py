#!/usr/bin/env python3
"""Sparse monitored QNN on all ten MNIST classes (small exact-state prototype).

Install: python -m pip install torch torchvision
Quick run: python mnist_sparse_qnn.py --epochs 5
Full data: python mnist_sparse_qnn.py --train-size 55000 --val-size 5000 --test-size 10000
Ablations: add --p-ent 0; or --p-meas 0; or --activation smooth
Offline check: python mnist_sparse_qnn.py --self-test

Architecture:
  fixed 28x28 -> 8x8 average pooling; trainable classical angle controllers;
  Ry(input encoding); repeated Ry(STE activation), Rz(trainable angle), sparse
  CZ matching, sparse projective Z measurements; outcomes feed the next layer;
  final Z expectations -> linear ten-class classifier.

Each layer's entanglers form a seeded random matching, retained with p_ent.
Measurement locations are fixed at initialization with per-site rate p_meas;
Born outcomes are freshly sampled for every image and trajectory. Seeds and
all architecture masks are saved in the checkpoint.

IMPORTANT GRADIENT/SCALING LIMITS:
  Forward trajectories and collapse are physical and normalized. Backward uses
  exact simulator autograd through gates/conditional normalized states, plus
  tanh-based STEs for binary activations and measurement feedback. The sampled
  projector is detached: derivatives of branch probabilities (score terms)
  are omitted. This is a BIASED heuristic, not an unbiased quantum gradient.
  Final Z readout uses exact conditional expectations, not finite-shot samples.
  A hardware implementation requires a separate gradient estimator; STE alone
  does not supply gate derivatives. The simulator costs O(2**n) memory per
  trajectory (more for autograd); start at 8 qubits. This experiment establishes
  neither classical hardness nor advantage, even if entanglement grows.

Outputs: history.csv, config.json, best.pt, test.json in --output.
Validation chooses the checkpoint; test data are evaluated only at the end.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


def ry(state, theta, q, n):
    # q=0 is the most significant computational-basis bit.
    v = state.reshape(-1, 2**q, 2, 2**(n-q-1))
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    c, s = torch.cos(theta/2)[:, None, None], torch.sin(theta/2)[:, None, None]
    return torch.stack((c*a-s*b, s*a+c*b), dim=2).reshape_as(state)


def activation(x, mode):
    soft = torch.tanh(x)
    if mode == 'smooth':
        return soft
    hard = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return soft + (hard-soft).detach()


class SparseQNN(nn.Module):
    def __init__(self, n=8, layers=4, p_ent=.35, p_meas=.08, seed=1962,
                 activation_mode='ste'):
        super().__init__()
        self.n, self.layers, self.activation_mode = n, layers, activation_mode
        self.encoder = nn.Linear(64, n)
        self.controllers = nn.ModuleList(nn.Linear(64+n, n) for _ in range(layers))
        self.rz_angles = nn.Parameter(.2*torch.randn(layers, n))
        self.readout = nn.Linear(n, 10)
        bits = ((torch.arange(2**n)[:, None] >> torch.arange(n-1, -1, -1)) & 1).float()
        self.register_buffer('bits', bits)
        self.register_buffer('z', 1-2*bits)
        rng = random.Random(seed)
        phases, measured = [], []
        self.edge_count = 0
        for _ in range(layers):
            order = list(range(n)); rng.shuffle(order)
            phase = torch.ones(2**n)
            for i in range(0, n-1, 2):
                if rng.random() < p_ent:
                    a, b = order[i:i+2]
                    phase *= 1-2*bits[:, a]*bits[:, b]
                    self.edge_count += 1
            phases.append(phase)
            measured.append([rng.random() < p_meas for _ in range(n)])
        self.register_buffer('cz_phases', torch.stack(phases))
        self.register_buffer('measured', torch.tensor(measured, dtype=torch.bool))

    def measure(self, state, q):
        p1 = (state.abs().square()*self.bits[:, q]).sum(-1).clamp(0, 1)
        outcome = (torch.rand_like(p1) < p1.detach()).float()
        # Project the physical forward state using the actual sampled bit.
        mask = (self.bits[:, q][None, :] == outcome[:, None]).to(state.dtype)
        projected = state*mask
        norm = projected.abs().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        state = projected/norm
        # Bias is intentional: replace derivative of the binary feedback by dp1.
        bit_ste = p1 + (outcome-p1).detach()
        return state, 2*bit_ste-1

    def forward(self, images, trajectories=1, return_state=False):
        x = F.adaptive_avg_pool2d(images, (8, 8)).flatten(1)*2-1
        batch = x.shape[0]
        x = x.repeat_interleave(trajectories, dim=0)
        state = torch.zeros(x.shape[0], 2**self.n, dtype=torch.complex64, device=x.device)
        state[:, 0] = 1
        encoding = math.pi*torch.tanh(self.encoder(x))
        for q in range(self.n):
            state = ry(state, encoding[:, q], q, self.n)
        memory = torch.zeros(x.shape[0], self.n, device=x.device)
        for layer, controller in enumerate(self.controllers):
            angles = (math.pi/3)*activation(controller(torch.cat((x, memory), -1)),
                                            self.activation_mode)
            for q in range(self.n):
                state = ry(state, angles[:, q], q, self.n)
            # Product of commuting one-qubit Rz gates, then the sparse CZ gates.
            phase = torch.exp(-.5j*(self.z @ self.rz_angles[layer]))
            state = state*phase*self.cz_phases[layer]
            feedback = []
            for q in range(self.n):
                if bool(self.measured[layer, q]):
                    state, bit = self.measure(state, q)
                    feedback.append(bit)
                else:
                    feedback.append(memory[:, q])
            memory = torch.stack(feedback, -1)
        features = (state.abs().square() @ self.z).reshape(batch, trajectories, self.n).mean(1)
        logits = self.readout(features)
        return (logits, state) if return_state else logits


@torch.no_grad()
def evaluate(model, loader, device, trajectories):
    model.eval()
    loss, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, trajectories)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    return {'loss': loss/count, 'accuracy': correct/count, 'examples': count}


@torch.no_grad()
def entropy_bits(state, n):
    singular = torch.linalg.svdvals(state.reshape(-1, 2**(n//2), 2**(n-n//2)))
    p = singular.square()
    return -(p*p.clamp_min(1e-12).log2()).sum(-1).mean().item()


def self_test():
    # Analytic Bell state checks both gates and projective measurement.
    m = SparseQNN(n=2, layers=1, p_ent=1, p_meas=1)
    v = torch.zeros(1, 4, dtype=torch.complex64); v[:, 0] = 1
    v = ry(v, torch.tensor([math.pi/2]), 0, 2)
    v = ry(v, torch.tensor([math.pi/2]), 1, 2)
    v = v*torch.tensor([1., 1., 1., -1.])
    assert abs(entropy_bits(v, 2)-1) < 1e-5
    v, _ = m.measure(v, 0)
    assert torch.allclose(v.abs().square().sum(-1), torch.ones(1), atol=1e-5)
    assert abs(entropy_bits(v, 2)) < 1e-5
    model = SparseQNN(n=4, layers=2, p_ent=1, p_meas=1)
    logits, state = model(torch.rand(3, 1, 28, 28), 2, return_state=True)
    assert logits.shape == (3, 10)
    assert torch.allclose(state.abs().square().sum(-1), torch.ones(6), atol=1e-5)
    F.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Before any mid-circuit measurement, compare autograd to finite differences.
    model = SparseQNN(n=3, layers=2, p_ent=1, p_meas=0, activation_mode='smooth')
    x = torch.rand(2, 1, 28, 28)
    model(x).square().sum().backward()
    parameter = model.encoder.weight
    analytic = parameter.grad[0, 0].item()
    with torch.no_grad():
        original = parameter[0, 0].item()
        parameter[0, 0] = original + .002
        plus = model(x).square().sum().item()
        parameter[0, 0] = original - .002
        minus = model(x).square().sum().item()
        parameter[0, 0] = original
    assert abs(analytic-(plus-minus)/.004) < .005
    print('Self-test passed: entanglement, collapse, normalization, gradients.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--qubits', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--p-ent', type=float, default=.35)
    p.add_argument('--p-meas', type=float, default=.08)
    p.add_argument('--activation', choices=['ste', 'smooth'], default='ste')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--train-size', type=int, default=5000)
    p.add_argument('--val-size', type=int, default=1000)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--train-trajectories', type=int, default=1)
    p.add_argument('--eval-trajectories', type=int, default=8)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=1962)
    p.add_argument('--device', default='auto', help='auto, cpu, or cuda')
    p.add_argument('--data', default='./mnist_data')
    p.add_argument('--output', default='./sparse_qnn_run')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.self_test:
        self_test(); return
    for key in ('qubits', 'layers', 'epochs', 'batch_size', 'train_size', 'val_size',
                'test_size', 'train_trajectories', 'eval_trajectories'):
        if getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    if args.qubits < 2 or not 0 <= args.p_ent <= 1 or not 0 <= args.p_meas <= 1 or args.lr <= 0:
        p.error('Use >=2 qubits, probabilities in [0,1], and positive lr.')
    if args.train_size+args.val_size > 60000 or args.test_size > 10000:
        p.error('MNIST contains 60000 training and 10000 test images.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'best.pt').exists():
        p.error('Output already contains best.pt; choose a new --output directory.')
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    train_data = MNIST(args.data, train=True, download=True, transform=ToTensor())
    test_data = MNIST(args.data, train=False, download=True, transform=ToTensor())
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(60000, generator=g).tolist()
    test_order = torch.randperm(10000, generator=g).tolist()
    train = DataLoader(Subset(train_data, order[:args.train_size]), batch_size=args.batch_size,
                       shuffle=True, generator=g)
    val = DataLoader(Subset(train_data, order[args.train_size:args.train_size+args.val_size]),
                     batch_size=args.batch_size)
    test = DataLoader(Subset(test_data, test_order[:args.test_size]), batch_size=args.batch_size)
    model = SparseQNN(args.qubits, args.layers, args.p_ent, args.p_meas,
                      args.seed, args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = dict(vars(args), torch_version=str(torch.__version__), resolved_device=str(device))
    (out/'config.json').write_text(json.dumps(config, indent=2))
    print(f'Device={device}; parameters={sum(p.numel() for p in model.parameters())}; '
          f'CZ gates={model.edge_count}; measurements={model.measured.sum().item()}', flush=True)
    print('Biased STE/conditional-state gradient; exact-state simulator. No hardness claim.', flush=True)
    best = -1.
    with (out/'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_accuracy',
                                             'val_loss', 'val_accuracy', 'seconds'])
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            model.train(); total_loss = correct = count = 0; start = time.time()
            for x, y in train:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, args.train_trajectories)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss; reduce learning rate.')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += loss.item()*y.numel()
                correct += (logits.argmax(-1) == y).sum().item(); count += y.numel()
            # Fixed validation randomness makes checkpoint comparisons less noisy.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
                torch.manual_seed(args.seed+10000)
                metrics = evaluate(model, val, device, args.eval_trajectories)
            row = dict(epoch=epoch, train_loss=total_loss/count, train_accuracy=correct/count,
                       val_loss=metrics['loss'], val_accuracy=metrics['accuracy'], seconds=time.time()-start)
            writer.writerow(row); f.flush()
            print(json.dumps(row), flush=True)
            if metrics['accuracy'] > best:
                best = metrics['accuracy']
                torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                            'val_accuracy': best}, out/'best.pt')
    checkpoint = torch.load(out/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    torch.manual_seed(args.seed+20000)
    result = evaluate(model, test, device, args.eval_trajectories)
    # A small final-trajectory entropy diagnostic; not a hardness certificate.
    images, _ = next(iter(val))
    with torch.no_grad():
        _, states = model(images[:16].to(device), 1, return_state=True)
        result['mean_final_halfcut_entropy_bits'] = entropy_bits(states, args.qubits)
    result['selected_epoch'] = checkpoint['epoch']
    (out/'test.json').write_text(json.dumps(result, indent=2))
    print('TEST:', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

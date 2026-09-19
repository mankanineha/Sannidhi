"""
CSI-FallNet: Dual-Modal Dilated Network with Cross-Modal Gating
Exact architecture from train_csi_fallnet.ipynb

Input:  (B, 500, 232)  — 500 time steps, 116 amp + 116 phase
Output: (B, 3)         — [fall_logit, motion_energy_logit, quality_logit]

Only logit[:,0] is used for fall detection:
    prob = sigmoid(logit[:,0])
    fall = prob >= threshold  (threshold stored in checkpoint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Depthwise-separable dilated causal conv with residual."""
    def __init__(self, c, d):
        super().__init__()
        self.pad = 2 * d
        self.dw = nn.Conv1d(c, c, 3, dilation=d, groups=c, bias=False)
        self.pw = nn.Conv1d(c, c, 1)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        r = x
        x = F.pad(x, (self.pad, 0))
        return r + F.gelu(self.bn(self.pw(self.dw(x))))


class CSIFallNet(nn.Module):
    """Dual-modal dilated causal network for WiFi CSI fall detection.

    Args:
        n_sub: subcarriers per modality (116)
        c:     channel width (48=student default)
        drop:  dropout rate
    """
    def __init__(self, n_sub=116, c=48, drop=0.08):
        super().__init__()
        self.n_sub = n_sub

        self.amp_proj   = nn.Conv1d(n_sub, c, 1)
        self.phase_proj = nn.Conv1d(n_sub, c, 1)

        self.backbone = nn.Sequential(
            *[Block(c, d) for d in (1, 2, 4, 8, 16, 32, 64, 128)]
        )

        self.gate_a = nn.Linear(c, c)
        self.gate_p = nn.Linear(c, c)

        self.vel_pw = nn.Conv1d(c, c, 1, bias=False)
        self.fuse   = nn.Sequential(nn.Conv1d(c * 2, c, 1, bias=False), nn.BatchNorm1d(c))

        self.score = nn.Conv1d(c, 1, 1)
        h = max(6, c // 8)
        self.gate  = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, c), nn.Sigmoid())

        self.head = nn.Sequential(
            nn.LayerNorm(c * 3),
            nn.Linear(c * 3, 32),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        amp   = x[:, :, :self.n_sub].transpose(1, 2)
        phase = x[:, :, self.n_sub:].transpose(1, 2)

        af = self.backbone(self.amp_proj(amp))
        pf = self.backbone(self.phase_proj(phase))

        ga = torch.sigmoid(self.gate_a(af.mean(-1))).unsqueeze(-1)
        gp = torch.sigmoid(self.gate_p(pf.mean(-1))).unsqueeze(-1)
        x  = ga * af + gp * pf

        vel = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        x   = F.gelu(self.fuse(torch.cat([x, self.vel_pw(vel)], 1)))

        a = (x * torch.softmax(self.score(x), -1)).sum(-1)
        m = x.amax(-1)
        l = x[:, :, -1]
        f = torch.cat([a, m, l], 1)
        g = torch.cat([self.gate(a), self.gate(m), self.gate(l)], 1)
        return self.head(f * g)

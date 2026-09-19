"""
CSISpectraNet: Dual-Stream Spectro-Temporal Network for WiFi CSI Fall Detection
Exact architecture from train_csi_spectranet.ipynb

Input:  (B, 500, 232)  — 500 time steps, 116 amp + 116 phase
Output: (B, 2)         — 2-class logits [no_fall, fall]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SUB = 116

class LearnedPhaseCorrection(nn.Module):
    def __init__(self, n_sub=116):
        super().__init__()
        self.n_sub = n_sub
        self.predictor = nn.Sequential(
            nn.Linear(n_sub, 48),
            nn.GELU(),
            nn.Linear(48, 2)
        )
        self.register_buffer('k_idx', torch.linspace(0, 1, n_sub).unsqueeze(0).unsqueeze(0))

    def forward(self, phase):
        params = self.predictor(phase)
        slope  = params[:, :, 0:1]
        offset = params[:, :, 1:2]
        correction = slope * self.k_idx + offset
        return phase - correction


class MotionAwareConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_raw  = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw_vel  = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw_acc  = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

    def forward(self, x):
        vel = x[:, :, 1:] - x[:, :, :-1]
        vel = F.pad(vel, (1, 0))
        acc = vel[:, :, 1:] - vel[:, :, :-1]
        acc = F.pad(acc, (1, 0))

        raw_feat = self.dw_raw(x)
        vel_feat = self.dw_vel(vel)
        acc_feat = self.dw_acc(acc)

        return self.mix(torch.cat([raw_feat, vel_feat, acc_feat], dim=1))


class FrequencyInteraction(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.filter_real = nn.Parameter(torch.ones(1, channels, n_freq))
        self.filter_imag = nn.Parameter(torch.zeros(1, channels, n_freq) * 0.01)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        X_freq = torch.fft.rfft(x, dim=-1)
        H = torch.complex(self.filter_real, self.filter_imag)
        Y_freq = X_freq * H
        y = torch.fft.irfft(Y_freq, n=x.size(-1), dim=-1)
        return residual + self.norm(y)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        need_proj = (in_ch != out_ch) or (stride != 1)
        self.skip    = nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False) if need_proj else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(out_ch) if need_proj else nn.Identity()

    def forward(self, x):
        identity = self.skip_bn(self.skip(x))
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class CrossModalAttention(nn.Module):
    def __init__(self, dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_a   = nn.Linear(dim, dim, bias=False)
        self.kv_p  = nn.Linear(dim, dim * 2, bias=False)

        self.q_p   = nn.Linear(dim, dim, bias=False)
        self.kv_a  = nn.Linear(dim, dim * 2, bias=False)

        self.norm_a = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, out_dim),
            nn.GELU()
        )
        self.out_norm = nn.BatchNorm1d(out_dim)
        self.attn_drop = nn.Dropout(dropout)

    def _cross_attend(self, q_proj, kv_proj, query, context):
        B, T, C = query.shape
        H, D = self.n_heads, self.head_dim

        q = q_proj(query).reshape(B, T, H, D).transpose(1, 2)
        kv = kv_proj(context).reshape(B, T, 2, H, D).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return out

    def forward(self, amp, phase):
        amp_t   = self.norm_a(amp.transpose(1, 2))
        phase_t = self.norm_p(phase.transpose(1, 2))

        amp_x_phase = self._cross_attend(self.q_a, self.kv_p, amp_t, phase_t)
        phase_x_amp = self._cross_attend(self.q_p, self.kv_a, phase_t, amp_t)

        combined = torch.cat([amp_x_phase, phase_x_amp], dim=-1)
        gates = self.gate(combined)
        gated = combined * gates

        fused = self.out_proj(gated)
        fused = self.out_norm(fused.transpose(1, 2))
        return fused


class TemporalFPN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels
        self.down2 = nn.Sequential(
            nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C), nn.GELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C), nn.GELU()
        )

        self.coarse_attn = nn.MultiheadAttention(C, num_heads=4, dropout=0.1, batch_first=True)
        self.coarse_ffn  = nn.Sequential(
            nn.Linear(C, C * 2), nn.GELU(), nn.Linear(C * 2, C)
        )
        self.coarse_norm1 = nn.LayerNorm(C)
        self.coarse_norm2 = nn.LayerNorm(C)

        self.lateral2 = nn.Conv1d(C, C, 1, bias=False)
        self.lateral1 = nn.Conv1d(C, C, 1, bias=False)

        self.smooth2 = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1, bias=False),
            nn.BatchNorm1d(C), nn.GELU()
        )
        self.smooth1 = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1, bias=False),
            nn.BatchNorm1d(C), nn.GELU()
        )

    def forward(self, x):
        p1 = x
        p2 = self.down2(p1)
        p3 = self.down3(p2)

        p3_t = p3.transpose(1, 2)
        p3_t = p3_t + self.coarse_attn(self.coarse_norm1(p3_t), self.coarse_norm1(p3_t), self.coarse_norm1(p3_t))[0]
        p3_t = p3_t + self.coarse_ffn(self.coarse_norm2(p3_t))
        p3   = p3_t.transpose(1, 2)

        p2_fused = self.lateral2(p2) + F.interpolate(p3, size=p2.size(-1), mode='nearest')
        p2_fused = self.smooth2(p2_fused)

        p1_fused = self.lateral1(p1) + F.interpolate(p2_fused, size=p1.size(-1), mode='nearest')
        p1_fused = self.smooth1(p1_fused)

        return p1_fused


class MultiPoolAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn_score = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        w = F.softmax(self.attn_score(x), dim=-1)
        attn = (x * w).sum(dim=-1)
        mx = x.amax(dim=-1)
        avg = x.mean(dim=-1)
        std = x.std(dim=-1)

        return torch.cat([attn, mx, avg, std], dim=1)


class CSISpectraNet(nn.Module):
    def __init__(self, n_sub=116, num_classes=2, dropout=0.2):
        super().__init__()
        self.n_sub = n_sub

        self.phase_correct = LearnedPhaseCorrection(n_sub)

        self.amp_embed   = nn.Sequential(nn.Conv1d(n_sub, 80, 1, bias=False), nn.BatchNorm1d(80), nn.GELU())
        self.phase_embed = nn.Sequential(nn.Conv1d(n_sub, 80, 1, bias=False), nn.BatchNorm1d(80), nn.GELU())

        self.amp_pos   = nn.Parameter(torch.randn(1, 80, 1) * 0.02)
        self.phase_pos = nn.Parameter(torch.randn(1, 80, 1) * 0.02)

        self.amp_motion   = MotionAwareConv(80)
        self.phase_motion = MotionAwareConv(80)

        self.amp_down1   = TemporalBlock(80, 96, stride=2)
        self.phase_down1 = TemporalBlock(80, 96, stride=2)

        self.amp_freq   = FrequencyInteraction(96, seq_len=250)
        self.phase_freq = FrequencyInteraction(96, seq_len=250)

        self.amp_down2   = TemporalBlock(96, 128, stride=2)
        self.phase_down2 = TemporalBlock(96, 128, stride=2)

        self.cross_modal = CrossModalAttention(dim=128, out_dim=192, n_heads=4, dropout=0.1)
        self.fpn = TemporalFPN(192)
        self.pool = MultiPoolAggregation(192)

        pool_dim = 192 * 4
        self.se = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 8),
            nn.GELU(),
            nn.Linear(pool_dim // 8, pool_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        amp_raw   = x[:, :, :self.n_sub]
        phase_raw = x[:, :, self.n_sub:]

        phase_clean = self.phase_correct(phase_raw)

        amp   = amp_raw.transpose(1, 2)
        phase = phase_clean.transpose(1, 2)

        amp   = self.amp_embed(amp) + self.amp_pos
        phase = self.phase_embed(phase) + self.phase_pos

        amp   = self.amp_motion(amp)
        phase = self.phase_motion(phase)

        amp   = self.amp_down1(amp)
        phase = self.phase_down1(phase)

        amp   = self.amp_freq(amp)
        phase = self.phase_freq(phase)

        amp   = self.amp_down2(amp)
        phase = self.phase_down2(phase)

        fused = self.cross_modal(amp, phase)
        fused = self.fpn(fused)
        pooled = self.pool(fused)
        pooled = pooled * self.se(pooled)

        return self.head(pooled)

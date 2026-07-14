"""BST4 — the vendored BST_CG_AP generalized to the 4 fixed court-quadrant streams.

Composes ONLY the vendored model classes (TCN, transformer encoders, cross-attention, MLP
heads) from worker/.cache/_vendor/BST @ BST_SHA — upstream's 2-player BST_* variants hard-cap
`n_people <= 2` and hard-code the 3-way stream split, so the 4-slot forward lives here, kept
line-for-line parallel to upstream where the semantics carry over (see bst.py:658-817).
What is genuinely new (and unit-tested): the slot-presence mask — absent players are all-zero
streams that would otherwise soak up AimPlayer weight and clamp the CleanGate min — and the
linear n-way AimPlayer normalization (upstream's alpha=(sim1-sim2+2)/4 is linear in the sims;
softmax would compress contrast and add a hidden temperature).

Unlike the rest of the worker, this module imports torch at the top: an nn.Module subclass
cannot defer it. Nothing on the light install imports this module (the torch section of
stroke/train.py and the guarded tests import it lazily).

Vendoring: one-time `git clone https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
worker/.cache/_vendor/BST && git -C worker/.cache/_vendor/BST checkout <BST_SHA>` — see
worker/training/bst_stroke_train.md. import_vendor() is collision-safe against
shuttle/tracknetv3.py, which imports its own vendored repo under the same top-level name
`model`: pre-existing sys.modules entries are snapshotted, replaced, and restored.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

_ROOT = Path(__file__).resolve().parents[3]
BST_REPO = os.environ.get("BST_REPO", str(_ROOT / "worker/.cache/_vendor/BST/stroke_classification"))
STROKE_MODEL = os.environ.get("STROKE_MODEL", str(_ROOT / "models/weights/stroke/bst_best.pt"))
BST_SHA = "fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df"

_VENDOR_NAMES = ("TCN", "FeedForward", "MLP", "MLP_Head", "TransformerEncoder",
                 "CrossTransformerLayer", "MultiHeadCrossAttention")
_VENDOR_CACHE: dict = {}


def import_vendor(repo: str = BST_REPO) -> dict:
    """Import the vendored model classes without clobbering anyone else's `model` module."""
    if not (Path(repo) / "model" / "bst.py").exists():
        raise RuntimeError(
            f"vendored BST not found at {repo} — clone "
            f"https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer at {BST_SHA} "
            f"(see worker/training/bst_stroke_train.md)")
    if _VENDOR_CACHE:
        return dict(_VENDOR_CACHE)
    displaced = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "model" or k.startswith("model.")}
    sys.path.insert(0, repo)
    try:
        bst = importlib.import_module("model.bst")
        tempose = importlib.import_module("model.tempose")
    finally:
        sys.path.remove(repo)
        for k in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
            del sys.modules[k]
        sys.modules.update(displaced)
    classes = {name: getattr(tempose if hasattr(tempose, name) else bst, name)
               for name in _VENDOR_NAMES}
    _VENDOR_CACHE.update(classes)
    return dict(classes)


class BST4(nn.Module):
    """4-slot BST: slots (near_left, near_right, far_left, far_right) share one pose TCN.

    variant="cg_ap": PPF + presence-masked AimPlayer + presence-masked CleanGate (the
    generalization of upstream BST_CG_AP). variant="bst0": upstream BST_0's unweighted tail
    (no PPF/AimPlayer/CleanGate modules at all; `pos` is accepted and ignored), kept for the
    sanctioned ablation. Head width d_model*(n_slots+1) either way.
    """

    VARIANTS = ("cg_ap", "bst0")

    def __init__(self, in_dim: int = 34, seq_len: int = 16, n_class: int = 7,
                 n_slots: int = 4, variant: str = "cg_ap",
                 d_model: int = 100, d_head: int = 128, n_head: int = 6,
                 depth_tem: int = 2, depth_inter: int = 1,
                 drop_p: float = 0.3, mlp_d_scale: int = 4, tcn_kernel_size: int = 5):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r} (one of {self.VARIANTS})")
        v = import_vendor()
        self.variant = variant
        self.n_slots = n_slots
        self.d_model = d_model

        if variant == "cg_ap":
            self.mlp_positions = v["MLP"](2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)
            self.mlp_clean = v["MLP"](d_model, d_model, d_model, drop_p)

        self.tcn_pose = v["TCN"](in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = v["TCN"](2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = v["TransformerEncoder"](
            d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = v["CrossTransformerLayer"](
            d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
        self.encoder_inter = v["TransformerEncoder"](
            d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        self.mlp_head = v["MLP_Head"](
            d_model * (n_slots + 1), n_class, d_model * mlp_d_scale, drop_p)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # verbatim upstream (bst.py init_weights): sinusoidal embeddings + ViT-style linears
        from positional_encodings.torch_encodings import PositionalEncoding1D

        p_enc = PositionalEncoding1D(self.d_model)
        self.embedding_tem.copy_(p_enc(self.embedding_tem))
        self.embedding_cross.copy_(p_enc(self.embedding_cross))
        self.embedding_inter.copy_(p_enc(self.embedding_inter))
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        self.apply(self._init_recursive)

    @staticmethod
    def _init_recursive(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    @staticmethod
    def slot_presence(joints: Tensor) -> Tensor:
        """(b, t, n, in_dim) -> (b, n) bool. A slot with all-zero joints is an absent player
        (exporter convention); PPF keeps its TCN input exactly zero (x*p + x = 0)."""
        return joints.abs().amax(dim=(1, 3)) > 0

    @staticmethod
    def aim_weights(sims: Tensor, present: Tensor) -> Tensor:
        """(b, n) cosine sims + presence -> (b, n) weights summing to 1. Linear in the sims —
        matches upstream's 2-player alpha at the endpoints and around equal sims."""
        w = (sims + 1.0) * present
        return w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def clean_input(cross_cls: Tensor, present: Tensor) -> Tensor:
        """(b, n, d) cross CLS tokens -> (b, d) elementwise min over PRESENT slots only
        (absent slots' near-constant tokens would clamp the min into a learned bias)."""
        masked = cross_cls.masked_fill(~present.unsqueeze(-1), torch.inf)
        out = masked.amin(dim=1)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    def forward(self, joints: Tensor, shuttle: Tensor, pos: Tensor) -> Tensor:
        # joints (b,t,n,in_dim), shuttle (b,t,2), pos (b,t,n,2) -> (b, n_class) logits.
        # No attention masks: windows are center-padded, zeros already encode missing data,
        # and an all-True prefix mask is arithmetically mask=None (see plan/spec).
        b, t, n, in_dim = joints.shape
        present = self.slot_presence(joints)

        x = joints.permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
        if self.variant == "cg_ap":
            pos_impact = self.mlp_positions(pos).permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
            x = x * pos_impact + x  # Pose Position Fusion
        x = self.tcn_pose(x)
        x = x.view(b, n, -1, t).transpose(-2, -1)               # (b, n, t, d)

        s = self.tcn_shuttle(shuttle.transpose(1, 2).contiguous())
        s = s.unsqueeze(1).transpose(-2, -1)                    # (b, 1, t, d)

        x = torch.cat((x, s), dim=1)                            # (b, m, t, d), m = n+1
        m, d = n + 1, x.shape[-1]
        cls_tem = self.learned_token_tem.view(1, 1, -1).expand(b * m, -1, -1)
        x = x.reshape(b * m, t, d)
        x = torch.cat((cls_tem, x), dim=1) + self.embedding_tem
        x = self.encoder_tem(self.pre_dropout(x))
        x = x.view(b, m, 1 + t, d)

        player_cls = x[:, :n, 0]                                # (b, n, d)
        shuttle_cls = x[:, n, 0]                                # (b, d)
        shuttle_tokens = x[:, n, 1:] + self.embedding_cross
        cls_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)

        cross_cls = []
        for i in range(n):  # one cross layer, called per slot — upstream calls it twice
            pi = self.cross_trans(x[:, i, 1:] + self.embedding_cross, shuttle_tokens)
            pi = torch.cat((cls_inter, pi), dim=1) + self.embedding_inter
            cross_cls.append(self.encoder_inter(pi)[:, 0])
        cross_cls = torch.stack(cross_cls, dim=1)               # (b, n, d)

        conclusions = player_cls + cross_cls
        if self.variant == "cg_ap":
            sims = nn.functional.cosine_similarity(
                cross_cls, shuttle_cls.unsqueeze(1).expand_as(cross_cls), dim=-1)
            conclusions = self.aim_weights(sims, present).unsqueeze(-1) * conclusions
            shuttle_cls = shuttle_cls - self.mlp_clean(self.clean_input(cross_cls, present))
        else:
            conclusions = present.unsqueeze(-1) * conclusions   # BST_0 tail: unweighted, absent zeroed

        return self.mlp_head(torch.cat((*conclusions.unbind(1), shuttle_cls), dim=1))


def save_checkpoint(path: Path, model: BST4, config: dict) -> None:
    """{"state_dict", "config"} — config must be BST4 kwargs (primitives only), so loading
    never unpickles vendored classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": dict(config)}, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[BST4, dict]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    model = BST4(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, dict(ckpt["config"])

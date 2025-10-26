"""Boltz2 backbone integration for FlowDock.

This module adapts components from the OM-TPS project (Boltz2) to provide a
score head that can be swapped into the FlowDock architecture.  The code is
largely derived from https://github.com/ASK-Berkeley/OM-TPS (commit
8fbc3a50af7dbb58872e6d89e638f8051d65dda6) and has been simplified and
extended to handle padded variable-length systems used in FlowDock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


def exists(val) -> bool:
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def center_zero(x: torch.Tensor) -> torch.Tensor:
    """Center coordinates to remove translation freedom."""

    return x - x.mean(dim=-2, keepdim=True)


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class GatedResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x, residual):
        gate = torch.sigmoid(self.gate(residual))
        return residual + gate * self.proj(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        edge_dim: Optional[int] = None,
        pos_emb: Optional[nn.Module] = None,
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.edge_proj = nn.Linear(edge_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.pos_emb = pos_emb

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = nodes.shape
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim=-1)
        e = self.edge_proj(edges)

        q = q.reshape(b, n, h, -1).transpose(1, 2)
        k = k.reshape(b, n, h, -1).transpose(1, 2)
        v = v.reshape(b, n, h, -1).transpose(1, 2)
        e = e.reshape(b, n, n, h, -1).permute(0, 3, 1, 2, 4)

        sim = torch.matmul(q, k.transpose(-2, -1))
        sim = sim * self.scale
        sim = sim + (e * self.scale).sum(-1)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            sim = sim.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(sim, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class List(nn.ModuleList):
    def append(self, module):  # type: ignore[override]
        super().append(module)


class GraphTransformerLucid(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        edge_dim: Optional[int] = None,
        heads: int = 8,
        gated_residual: bool = True,
        with_feedforwards: bool = False,
        norm_edges: bool = False,
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)
        self.layers = nn.ModuleList()
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        for _ in range(depth):
            attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, edge_dim=edge_dim))
            attn_residual = GatedResidual(dim) if gated_residual else Residual()
            if with_feedforwards:
                ff = PreNorm(dim, FeedForward(dim))
                ff_residual = GatedResidual(dim) if gated_residual else Residual()
            else:
                ff = None
                ff_residual = None
            self.layers.append(nn.ModuleList([attn, attn_residual, ff, ff_residual]))

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = self.norm_edges(edges)
        for attn, attn_residual, ff, ff_residual in self.layers:
            nodes = attn_residual(attn(nodes, edges, mask=mask), nodes)
            if exists(ff):
                nodes = ff_residual(ff(nodes), nodes)
        return nodes, edges


def compute_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,
        inputs=positions,
        grad_outputs=torch.ones_like(energy),
        retain_graph=training,
        create_graph=training,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        raise RuntimeError("Boltz2 backbone returned no gradients for energy.")
    return -gradient


class GraphTransformer(nn.Module):
    """Graph transformer adapted from Boltz2 with padding-aware masking."""

    def __init__(
        self,
        num_beads: int,
        hidden_nf: int,
        device: Optional[torch.device] = None,
        n_layers: int = 4,
        use_intrinsic_coords: bool = False,
        use_abs_coords: bool = True,
        use_distances: bool = True,
        conservative: bool = True,
        use_bead_identities: bool = False,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.use_intrinsic_coords = use_intrinsic_coords
        self.use_distances = use_distances
        self.use_abs_coords = use_abs_coords
        self.conservative = conservative
        self.use_bead_identities = use_bead_identities
        self.heads = heads
        self.num_beads = num_beads

        in_node_nf = num_beads + 1 + (3 if use_abs_coords else 0)
        if use_bead_identities:
            in_node_nf += hidden_nf
        in_edge_nf = (
            3 * use_intrinsic_coords
            + (1 if use_distances else 0)
            + (1 if (not use_intrinsic_coords and not use_distances) else 0)
        )

        self.node_embedding = nn.Linear(in_node_nf, hidden_nf)
        self.edge_embedding = nn.Linear(in_edge_nf, hidden_nf)

        if self.conservative:
            self.node_decoder = nn.Linear(hidden_nf, 1)
        else:
            self.node_decoder = nn.Linear(hidden_nf, 3)

        if self.use_bead_identities:
            self.bead_embedding = nn.Embedding(40, hidden_nf)

        self.graphtransformer = GraphTransformerLucid(
            dim=hidden_nf,
            dim_head=dim_head,
            depth=n_layers,
            edge_dim=hidden_nf,
            with_feedforwards=True,
            gated_residual=True,
            heads=self.heads,
        )

        self.to(self.device)

    def get_edge_attr(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_distances and not self.use_intrinsic_coords:
            diff = x.unsqueeze(1) - x.unsqueeze(2)
            dist = torch.sum(diff**2, dim=3, keepdim=True)
            return dist
        if self.use_intrinsic_coords and not self.use_distances:
            return x.unsqueeze(1) - x.unsqueeze(2)
        if self.use_intrinsic_coords and self.use_distances:
            diff = x.unsqueeze(1) - x.unsqueeze(2)
            dist = torch.sum(diff**2, dim=3, keepdim=True)
            return torch.cat([diff, dist], dim=3)
        bs, n_nodes, _ = x.size()
        return torch.zeros(bs, n_nodes, n_nodes, 1, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor],
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        return_energy: bool = False,
    ) -> torch.Tensor:
        x = center_zero(x)
        x = x.requires_grad_(self.conservative)
        bs, n_nodes, _ = x.shape

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=x.device)
        if t.dim() == 1:
            t = t[:, None, None]
        if t.shape[0] == 1:
            t = t.repeat(bs, 1, 1)
        t = t.to(x.dtype)

        if h is None:
            h = torch.eye(n_nodes, device=x.device)
        if h.dim() == 2:
            h = h.unsqueeze(0).repeat(bs, 1, 1)

        if mask is None:
            mask = torch.ones(bs, n_nodes, dtype=torch.bool, device=x.device)

        if z is not None and self.use_bead_identities:
            if z.dim() == 1:
                z = z.unsqueeze(0).repeat(bs, 1)
            if z.shape[0] != h.shape[0]:
                z = z.repeat(h.shape[0] // z.shape[0], 1)
            padding_idx = z == 0
            z = self.bead_embedding(z)
            h = torch.cat((h, z), dim=2)
            mask = mask & (~padding_idx)

        if self.use_abs_coords:
            nodes = torch.cat((h, x, t.repeat(1, n_nodes, 1)), dim=2)
        else:
            nodes = torch.cat((h, t.repeat(1, n_nodes, 1)), dim=2)

        edge_attr = self.get_edge_attr(x)
        edge_attr = self.edge_embedding(edge_attr)

        nodes, _ = self.graphtransformer(nodes, edge_attr, mask=mask)
        output = self.node_decoder(nodes)

        if self.conservative:
            energy = output
            if return_energy:
                return energy
            forces = compute_forces(energy, x, self.training)
        else:
            forces = output

        forces = forces * mask.unsqueeze(-1)
        return forces


class TruncatedAction(nn.Module):
    """Simplified Onsager-Machlup action used for Boltz2 training."""

    def __init__(self, gamma: float, dt: float):
        super().__init__()
        self.gamma = gamma
        self.dt = dt

    def forward(
        self,
        path: torch.Tensor,
        forces: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # path: (B, K, N, 3), forces: (B, K-1, N, 3), mask: (B, N)
        disp = path[:, 1:] - path[:, :-1]
        path_term = (disp**2).sum(dim=-1) / (2 * self.dt)
        force_term = (forces**2).sum(dim=-1) * (self.dt / (2 * self.gamma**2))
        mask = mask.unsqueeze(1).to(path_term.dtype)
        path_term = path_term * mask
        force_term = force_term * mask
        return path_term.sum(dim=(1, 2)), force_term.sum(dim=(1, 2))


@dataclass
class Boltz2Config:
    max_protein_atoms: int
    max_ligand_atoms: int
    hidden_nf: int
    n_layers: int
    heads: int
    dim_head: int
    use_intrinsic_coords: bool
    use_abs_coords: bool
    use_distances: bool
    conservative: bool
    use_bead_identities: bool
    integration_dt: float
    integration_steps: int
    gamma: float


class Boltz2ScoreHead(nn.Module):
    """FlowDock score head that delegates to the Boltz2 backbone."""

    def __init__(
        self,
        protein_model_cfg: Dict,
        score_cfg: Dict,
        task_cfg: Dict,
    ):
        super().__init__()
        if "boltz2" not in score_cfg:
            raise ValueError("Boltz2 backend requires a `boltz2` configuration block.")

        cfg = score_cfg.boltz2
        self.config = Boltz2Config(
            max_protein_atoms=cfg.max_protein_atoms,
            max_ligand_atoms=cfg.max_ligand_atoms,
            hidden_nf=cfg.hidden_nf,
            n_layers=cfg.n_layers,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            use_intrinsic_coords=cfg.use_intrinsic_coords,
            use_abs_coords=cfg.use_abs_coords,
            use_distances=cfg.use_distances,
            conservative=cfg.conservative,
            use_bead_identities=cfg.use_bead_identities,
            integration_dt=cfg.integration_dt,
            integration_steps=cfg.integration_steps,
            gamma=cfg.gamma,
        )

        num_beads = cfg.max_protein_atoms + cfg.max_ligand_atoms
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = GraphTransformer(
            num_beads=num_beads,
            hidden_nf=cfg.hidden_nf,
            device=device,
            n_layers=cfg.n_layers,
            use_intrinsic_coords=cfg.use_intrinsic_coords,
            use_abs_coords=cfg.use_abs_coords,
            use_distances=cfg.use_distances,
            conservative=cfg.conservative,
            use_bead_identities=cfg.use_bead_identities,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
        )

        self.register_buffer("base_node_features", torch.eye(num_beads), persistent=False)
        self.action = TruncatedAction(gamma=cfg.gamma, dt=cfg.integration_dt)
        self.num_beads = num_beads
        self.score_cfg = score_cfg
        self.task_cfg = task_cfg

    @property
    def protein_slice(self) -> slice:
        return slice(0, self.config.max_protein_atoms)

    @property
    def ligand_slice(self) -> slice:
        return slice(
            self.config.max_protein_atoms,
            self.config.max_protein_atoms + self.config.max_ligand_atoms,
        )

    def forward(
        self,
        batch: Dict,
        frozen_lig: bool = False,
        frozen_prot: bool = False,
        **_: Dict,
    ) -> Dict[str, Optional[torch.Tensor]]:
        device = self.base_node_features.device
        features = batch["features"]
        metadata = batch["metadata"]
        batch_size = metadata["num_structid"]

        res_atom_mask = features["res_atom_mask"].bool()
        protein_coords_padded = features["input_protein_coords"].to(device)

        max_prot_atoms = self.config.max_protein_atoms
        coords = torch.zeros(batch_size, self.num_beads, 3, device=device)
        mask = torch.zeros(batch_size, self.num_beads, dtype=torch.bool, device=device)

        prot_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        protein_coords_flat = []
        for i in range(batch_size):
            prot_coords_sample = protein_coords_padded[i][res_atom_mask[i]]
            n_prot = prot_coords_sample.shape[0]
            prot_counts[i] = n_prot
            if n_prot > max_prot_atoms:
                raise ValueError(
                    "Boltz2 configuration does not allocate enough protein atoms "
                    "to cover the current batch. Increase `max_protein_atoms`."
                )
            coords[i, self.protein_slice.start : self.protein_slice.start + n_prot] = (
                prot_coords_sample
            )
            mask[i, self.protein_slice.start : self.protein_slice.start + n_prot] = True
            protein_coords_flat.append(prot_coords_sample)

        ligand_only = batch["misc"].get("protein_only", False)
        if not ligand_only:
            lig_coords_padded = features["input_ligand_coords"].to(device)
            lig_counts_raw = metadata.get("num_i_per_sample")
            if lig_counts_raw is None:
                lig_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif isinstance(lig_counts_raw, torch.Tensor):
                lig_counts = lig_counts_raw.to(device).long()
            else:
                lig_counts = torch.tensor(lig_counts_raw, device=device, dtype=torch.long)
            max_lig_atoms = self.config.max_ligand_atoms
            ligand_slice = self.ligand_slice
            for i in range(batch_size):
                n_lig = int(lig_counts[i].item())
                if n_lig > max_lig_atoms:
                    raise ValueError(
                        "Boltz2 configuration does not allocate enough ligand atoms "
                        "to cover the current batch. Increase `max_ligand_atoms`."
                    )
                if n_lig > 0:
                    coords[i, ligand_slice.start : ligand_slice.start + n_lig] = (
                        lig_coords_padded[i, :n_lig]
                    )
                    mask[i, ligand_slice.start : ligand_slice.start + n_lig] = True
        else:
            lig_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
            lig_coords_padded = None
            ligand_slice = self.ligand_slice

        timestep_encoding = features.get("timestep_encoding_prot")
        if timestep_encoding is None:
            t = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            t = timestep_encoding.view(batch_size, -1).mean(dim=1).to(torch.float32)

        coords = coords.to(self.base_node_features.dtype)
        mask = mask.to(coords.device)

        node_features = self.base_node_features[: self.num_beads, : self.num_beads]
        forces = self.backbone(coords, node_features, t, mask=mask)

        path = [coords]
        coords_next = coords
        forces_history = []
        forces_current = forces
        for step in range(self.config.integration_steps):
            if step > 0:
                forces_current = self.backbone(coords_next, node_features, t, mask=mask)
            forces_history.append(forces_current)
            coords_next = coords_next + self.config.integration_dt * forces_current
            path.append(coords_next)

        path_tensor = torch.stack(path, dim=1)
        forces_path = torch.stack(forces_history, dim=1)
        path_term, force_term = self.action(path_tensor, forces_path, mask)
        total_action = path_term + force_term

        final_protein_coords_padded = protein_coords_padded.clone()
        final_protein_coords_flat = []
        for i in range(batch_size):
            n_prot = int(prot_counts[i].item())
            prot_slice = slice(self.protein_slice.start, self.protein_slice.start + n_prot)
            updated = coords_next[i, prot_slice]
            final_protein_coords_flat.append(updated)
            final_protein_coords_padded[i][res_atom_mask[i]] = updated

        final_protein_coords_flat_tensor = (
            torch.cat(final_protein_coords_flat, dim=0)
            if final_protein_coords_flat
            else torch.zeros(0, 3, device=device)
        )

        if not ligand_only:
            final_lig_coords_flat = []
            final_lig_coords = lig_coords_padded.clone() if lig_coords_padded is not None else None
            for i in range(batch_size):
                n_lig = int(lig_counts[i].item())
                if n_lig == 0:
                    continue
                lig_slice = slice(ligand_slice.start, ligand_slice.start + n_lig)
                updated = coords_next[i, lig_slice]
                if final_lig_coords is not None:
                    final_lig_coords[i, :n_lig] = updated
                final_lig_coords_flat.append(updated)
            final_lig_coords_flat_tensor = (
                torch.cat(final_lig_coords_flat, dim=0)
                if final_lig_coords_flat
                else torch.zeros(0, 3, device=device)
            )
        else:
            final_lig_coords_flat_tensor = None

        ret: Dict[str, Optional[torch.Tensor]] = {
            "final_embedding_prot_atom": None,
            "final_embedding_prot_res": None,
            "final_embedding_lig_atom": None,
            "final_coords_prot_atom": final_protein_coords_flat_tensor,
            "final_coords_prot_atom_padded": final_protein_coords_padded,
            "final_coords_lig_atom": final_lig_coords_flat_tensor,
            "boltz2_action": {
                "path": path_term,
                "force": force_term,
                "total": total_action,
            },
        }

        return ret


def resolve_boltz2_score_head(
    protein_model_cfg: Dict,
    score_cfg: Dict,
    task_cfg: Dict,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    del state_dict  # Unused but kept for API parity
    return Boltz2ScoreHead(protein_model_cfg, score_cfg, task_cfg)


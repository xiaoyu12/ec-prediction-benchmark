"""Model definitions for the three released enzyme-MIL checkpoints.

Each enzyme is a *bag* and its DCTdomain domains are *instances*. A gated-attention pooler turns
the bag into one fixed-width representation, and a classifier head emits independent logits over a
fixed vocabulary of 5,242 EC numbers.

Two architectures live here because the three checkpoints come from two training generations:

``AttentionMIL``
    ``milbc`` and ``milbc_plus``. Reads the local domain tokens. ``milbc_plus`` concatenates the
    whole-sequence ESM2 embedding onto the pooled bag representation before the classifier.

``JointMILCL``
    ``milcl_plus``. Reads the same tokens plus the whole-protein DCT vector as a seventeenth,
    fuses ESM2 into every token before pooling, and carries a 256-dimensional contrastive
    projection alongside the classifier. Max-separation inference compares that projection against
    the split100 EC centroids in ``checkpoints/milcl_plus/centroids.npy``.

``ESM2BC``
    ``esm2_bc``, the paper's sequence-level baseline (no released checkpoint; train it with
    ``train.py``). The MIL-BC classifier head applied to the whole-sequence ESM2 embedding alone, so
    the comparison with ``milbc_plus`` isolates what the domain bag adds.

The geometry of each checkpoint is fixed; ``load_state_dict`` runs strict, so a mismatch raises
rather than loading silently wrong weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

NUM_CLASSES = 5242
DOMAIN_DIM = 480
ESM_DIM = 1280

# Geometry of the MILCL backbone, taken from the MILBC+ checkpoint it was initialised from.
MILCL_CONFIG = {
    "instance_hidden_dim": 512,
    "attention_hidden_dim": 256,
    "esm_hidden_dim": 512,
    "classifier_hidden_dim": 512,
    "projection_hidden_dim": 512,
    "projection_dim": 256,
    "dropout": 0.2,
    "projection_dropout": 0.1,
}


class AttentionMIL(nn.Module):
    """Gated-attention MIL over 16 local domain tokens.

    ``global_input_dim == 0`` is the MILBC form. A positive value is the MILBC+ form: the
    whole-sequence ESM2 embedding is concatenated onto the pooled bag representation, so the
    domain branch and the sequence branch stay independent until the classifier.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        instance_input_dim: int = DOMAIN_DIM,
        instance_embedding_dim: int = 480,
        attention_hidden_dim: int = 320,
        global_input_dim: int = 0,
        classifier_hidden_dim: int = 320,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.global_input_dim = global_input_dim
        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_input_dim, instance_embedding_dim),
            nn.LayerNorm(instance_embedding_dim),
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(instance_embedding_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.Tanh(),
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(instance_embedding_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(instance_embedding_dim + global_input_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        instances_in_bag: Tensor,
        mask: Tensor | None = None,
        global_feature: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        h_k = self.instance_encoder(instances_in_bag)
        u_k = self.attention_V(h_k) * self.attention_gate(h_k)
        e_k = self.attention_w(u_k).squeeze(-1)
        if mask is not None:
            e_k = e_k.masked_fill(mask == 0, -1e9)
        alpha_k = torch.softmax(e_k, dim=1)
        bag_representation = torch.sum(alpha_k.unsqueeze(-1) * h_k, dim=1)
        if global_feature is not None:
            bag_representation = torch.cat((bag_representation, global_feature), dim=1)
        return self.classifier(bag_representation), bag_representation


class ESM2BC(nn.Module):
    """ESM2-BC: the ``AttentionMIL`` classifier head on the whole-sequence ESM2 embedding alone.

    ``forward`` takes the same arguments as ``AttentionMIL`` so the two are interchangeable in
    ``infer.py`` and ``train.py``; the domain tokens and mask are ignored.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_dim: int = ESM_DIM,
        classifier_hidden_dim: int = 320,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        instances_in_bag: Tensor | None = None,
        mask: Tensor | None = None,
        global_feature: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if global_feature is None:
            raise ValueError("ESM2-BC needs the whole-sequence ESM2 embedding")
        return self.classifier(global_feature), global_feature


def ec_hierarchy(classes: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """EC tree over the vocabulary: ``(nodes, parent, level, leaf_node)``.

    ``nodes`` lists every prefix ``a``, ``a.b``, ``a.b.c`` and every full EC, ordered by level so a
    parent always precedes its children; ``parent[i]`` is the parent's position (-1 at level 1);
    ``level[i]`` is 1..4; ``leaf_node[j]`` is the node of ``classes[j]``.
    """
    levels: list[set[str]] = [set(), set(), set(), set()]
    for ec in classes:
        parts = ec.split(".")
        if len(parts) != 4:
            raise ValueError(f"EC vocabulary entries must have four levels, found {ec}")
        for depth in range(4):
            levels[depth].add(".".join(parts[: depth + 1]))
    nodes = [node for depth in range(4) for node in sorted(levels[depth])]
    position = {node: i for i, node in enumerate(nodes)}
    parent = np.array([position[n.rsplit(".", 1)[0]] if "." in n else -1 for n in nodes], dtype=np.int64)
    level = np.array([n.count(".") + 1 for n in nodes], dtype=np.int64)
    leaf_node = np.array([position[ec] for ec in classes], dtype=np.int64)
    return nodes, parent, level, leaf_node


class DomainAttentionEC(nn.Module):
    """ESM2-BC plus label-specific attention over per-domain ESM2 embeddings.

    Two branches whose logits are summed per output:

    * global: the ``ESM2BC`` network on the whole-sequence ESM2 embedding;
    * domain: every output ``k`` has a learned query ``q_k``; over the protein's domain features
      ``r_m`` (mean-pooled ESM2 residue embeddings of each DCTdomain domain)
      ``alpha_km = softmax_m(q_k . W_K r_m / sqrt(d))``, ``z_k = sum_m alpha_km W_V r_m``, and the
      domain logit is ``w_k . dropout(z_k)`` with a learned readout ``w_k``.

    ``hierarchical=False``: the outputs are the 5,242 EC numbers (independent sigmoids, as ESM2-BC).
    ``hierarchical=True``: the outputs are every node of the EC tree (``ec_hierarchy``) and are
    *conditional* logits, ``P(k | x) = P(parent(k) | x) * sigmoid(logit_k)``;
    ``leaf_logits`` turns them into one logit per EC with ``sigmoid(leaf logit) = P(EC | x)``, so the
    usual ``sigmoid >= 0.5`` rule (``infer.call_ec``) applies unchanged.
    """

    def __init__(
        self,
        classes: list[str],
        domain_dim: int = ESM_DIM,
        global_dim: int = ESM_DIM,
        width: int = 64,
        hierarchical: bool = True,
        classifier_hidden_dim: int = 320,
        dropout_rate: float = 0.2,
        use_domains: bool = True,
    ) -> None:
        super().__init__()
        self.hierarchical = hierarchical
        self.use_domains = use_domains  # False: global branch only (ablation: ESM2-BC, +hierarchy if set)
        if hierarchical:
            nodes, parent, level, leaf_node = ec_hierarchy(list(classes))
        else:
            nodes, parent = list(classes), np.full(len(classes), -1)
            level, leaf_node = np.full(len(classes), 4), np.arange(len(classes))
        self.nodes = nodes
        self.register_buffer("parent", torch.from_numpy(parent))
        self.register_buffer("level", torch.from_numpy(level))
        self.register_buffer("leaf_node", torch.from_numpy(leaf_node))
        n_out = len(nodes)
        self.global_branch = ESM2BC(n_out, global_dim, classifier_hidden_dim, dropout_rate)
        self.key = nn.Linear(domain_dim, width, bias=False)
        self.value = nn.Linear(domain_dim, width)
        self.query = nn.Parameter(torch.randn(n_out, width) * 0.02)
        self.readout = nn.Parameter(torch.randn(n_out, width) * 0.02)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, domains: Tensor, mask: Tensor, global_feature: Tensor) -> tuple[Tensor, Tensor]:
        """Output logits (conditional ones when hierarchical) and the attention ``(batch, outputs, domains)``."""
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every protein needs at least one domain")
        global_logits = self.global_branch(global_feature=global_feature)[0]
        if not self.use_domains:
            return global_logits, mask.new_zeros(mask.shape[0], 0, mask.shape[1], dtype=global_logits.dtype)
        keys = self.key(domains)
        values = self.value(domains)
        scores = torch.einsum("bmd,cd->bcm", keys, self.query) / keys.size(-1) ** 0.5
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        attention = scores.softmax(dim=-1)
        context = torch.einsum("bcm,bmd->bcd", attention, values)
        domain_logits = (self.dropout(context) * self.readout[None]).sum(dim=-1)
        return domain_logits + global_logits, attention

    def node_log_probs(self, logits: Tensor) -> Tensor:
        """``log P(node | x)`` for every output: the conditional log-sigmoids summed down the tree."""
        log_prob = F.logsigmoid(logits)
        if not self.hierarchical:
            return log_prob
        out = log_prob.clone()
        for depth in range(2, 5):
            nodes = torch.nonzero(self.level == depth).squeeze(1)
            out[:, nodes] = log_prob[:, nodes] + out[:, self.parent[nodes]]
        return out

    def leaf_logits(self, logits: Tensor) -> Tensor:
        """One logit per EC of the vocabulary, ``sigmoid(leaf logit) = P(EC | x)``."""
        if not self.hierarchical:
            return logits[:, self.leaf_node]
        log_p = self.node_log_probs(logits)[:, self.leaf_node].clamp(max=-1e-7)
        # log(1 - p) from log p, stably: log(-expm1(log p)) near p = 1, log1p(-exp(log p)) otherwise
        log_not = torch.where(log_p > -0.6931, torch.log(-torch.expm1(log_p)), torch.log1p(-torch.exp(log_p)))
        return log_p - log_not


class GatedAttentionPooler(nn.Module):
    """Gated attention over the 17 domain tokens, concatenated with their mean and max."""

    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim * 3
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(f"tokens must have shape (batch, instances, {self.input_dim})")
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("domain mask must be boolean with shape (batch, instances)")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each bag must contain at least one valid domain")
        encoded = self.encoder(self.input_norm(tokens))
        gated = torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        scores = self.attention_w(gated).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        mask_values = mask.unsqueeze(-1)
        mean = (encoded * mask_values).sum(dim=1) / mask_values.sum(dim=1)
        maximum = encoded.masked_fill(~mask_values, torch.finfo(encoded.dtype).min).max(dim=1).values
        return torch.cat((pooled, mean, maximum), dim=-1), attention


class MILBCPlusBackbone(nn.Module):
    """The MILCL backbone: ESM2 is encoded once and fused into every domain token before pooling."""

    def __init__(self) -> None:
        super().__init__()
        config = MILCL_CONFIG
        self.esm_encoder = nn.Sequential(
            nn.LayerNorm(ESM_DIM),
            nn.Linear(ESM_DIM, config["esm_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )
        self.pooler = GatedAttentionPooler(
            DOMAIN_DIM + config["esm_hidden_dim"],
            config["instance_hidden_dim"],
            config["attention_hidden_dim"],
            config["dropout"],
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.pooler.output_dim),
            nn.Linear(self.pooler.output_dim, config["classifier_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["classifier_hidden_dim"], NUM_CLASSES),
        )

    def encode(self, domains: Tensor, domain_mask: Tensor, esm2: Tensor) -> tuple[Tensor, Tensor]:
        if esm2.ndim != 2 or esm2.shape[0] != domains.shape[0] or esm2.shape[1] != ESM_DIM:
            raise ValueError(f"esm2 must have shape (batch, {ESM_DIM})")
        esm_representation = self.esm_encoder(esm2)
        repeated = esm_representation.unsqueeze(1).expand(-1, domains.shape[1], -1)
        representation, attention = self.pooler(torch.cat((domains, repeated), dim=-1), domain_mask)
        return representation, attention

    def classify(self, representation: Tensor) -> Tensor:
        return self.classifier(representation)


class JointProjection(nn.Sequential):
    """Contrastive head. Its output is what max-separation compares against the EC centroids."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class JointMILCL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = MILBCPlusBackbone()
        self.projection = JointProjection(
            self.backbone.pooler.output_dim,
            MILCL_CONFIG["projection_hidden_dim"],
            MILCL_CONFIG["projection_dim"],
            MILCL_CONFIG["projection_dropout"],
        )

    def forward(self, domains: Tensor, domain_mask: Tensor, esm2: Tensor) -> Tensor:
        """The 256-dimensional contrastive embedding of every enzyme in the batch."""
        representation, _ = self.backbone.encode(domains, domain_mask, esm2)
        return self.projection(representation)


def load_milbc(checkpoint: str, with_esm: bool, device: str = "cpu") -> AttentionMIL:
    """Rebuild ``milbc`` (``with_esm=False``) or ``milbc_plus`` (``with_esm=True``)."""
    model = AttentionMIL(global_input_dim=ESM_DIM if with_esm else 0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_esm2bc(checkpoint: str, device: str = "cpu") -> ESM2BC:
    """Rebuild ``esm2_bc`` from a ``train.py`` checkpoint."""
    model = ESM2BC()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_milcl(checkpoint: str, centroids: str, device: str = "cpu") -> tuple[JointMILCL, np.ndarray]:
    """Rebuild ``milcl_plus`` and read the EC centroids max-separation scores against."""
    model = JointMILCL()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    centers = np.load(centroids, allow_pickle=False)
    if centers.shape != (NUM_CLASSES, MILCL_CONFIG["projection_dim"]):
        raise ValueError(f"centroids must be ({NUM_CLASSES}, {MILCL_CONFIG['projection_dim']}), found {centers.shape}")
    return model.eval().to(device), centers

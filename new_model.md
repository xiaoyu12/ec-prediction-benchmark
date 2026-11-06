Implement another attention model using the domain annotations.

For this model, we will combine two model branches, one is the baseline ESM2-BC model, the other is based on the attention over domain features. 

**Global dense ESM embedding + attention over domain features**


To implement the attention branch.

- The domains are computed same way as the mil-bc method [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/429683).
- Domain feature $r_m$ can be computed as the mean-pooled ESM2 embeddings of residues in the domain. 
- **Learn label-specific attention.** Give each of EC label a learned query. Each query assigns different weights to the domain:

   $$
   \alpha_{km}=
   \operatorname{softmax}_{m}
   \left(\frac{q_k^\top W_Kr_m}{\sqrt d}\right),
   \qquad
   z_k=\sum_m\alpha_{km}W_Vr_m.
   $$

   The classifier for label \(k\) uses \(z_k\), together with the global dense embedding.

This is a form of multiple-instance learning: the protein has a collection of domains whose evidence is aggregated into protein-level predictions. 

A minimal PyTorch attention head looks like (change region to domain):

```python
import torch
from torch import nn

class RegionAttentionHead(nn.Module):
    def __init__(self, region_dim, dense_dim, width=64, n_labels=52):
        super().__init__()
        self.key = nn.Linear(region_dim, width, bias=False)
        self.value = nn.Linear(region_dim, width)
        self.query = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.readout = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.global_head = nn.Linear(dense_dim, n_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, regions, valid, global_embedding):
        # regions: [batch, regions, region_dim], padded with finite zeros
        # valid:   [batch, regions], boolean; excludes padded regions
        # global_embedding: [batch, dense_dim]
        if not valid.any(dim=1).all():
            raise ValueError("Every protein needs at least one valid region")

        keys = self.key(regions)
        values = self.value(regions)

        scores = torch.einsum("bmd,cd->bcm", keys, self.query)
        scores = scores / keys.size(-1) ** 0.5
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        attention = scores.softmax(dim=-1)

        context = torch.einsum("bcm,bmd->bcd", attention, values)
        regional_logits = (
            self.dropout(context) * self.readout[None, :, :]
        ).sum(dim=-1)

        logits = regional_logits + self.global_head(global_embedding)
        return logits, attention
```

Furthermore, apply the hierarchy structure of EC labels:

$$
P(\text{subcategory }k\mid x)
=
P(\text{parent}(k)\mid x)\,
P(k\mid \text{parent}(k),x).
$$

Train the conditional child heads using examples whose parent is known positive, masking unresolved child annotations. 


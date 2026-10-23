import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Any
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from main.dexhand_spec import ObservationLayout, get_hand_spec
from main.dataset.semantic_contact import (
    ACTUAL_CONTACT,
    BLACK_PROXIMITY,
    BLACK_VIOLATION,
    CURRICULUM_WEIGHT,
    EXPECTED_CONTACT,
    RED_CONTACT,
    RED_PROXIMITY,
)


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        B, SeqLen, D = query.shape

        qkv = self.in_proj(query)
        qkv = qkv.view(B, SeqLen, 3, self.num_heads, self.head_dim)
        
        # Permute to [B, H, S, HeadDim]
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        
        if attn_mask is not None and attn_mask.dim() == 3:
             attn_mask = attn_mask.view(B, self.num_heads, SeqLen, SeqLen)

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout.p if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, SeqLen, D)
        return self.out_proj(out), None


class SemanticConditionEncoder(nn.Module):
    """Encode six uncompressed semantic relations for the five RH contact links."""

    NUM_CONTACT_LINKS = 5
    SEMANTIC_CONDITION_DIM = 6

    def __init__(
        self,
        d_model: int,
        finger_embedding_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.finger_embedding = nn.Embedding(
            self.NUM_CONTACT_LINKS, finger_embedding_dim
        )
        self.net = nn.Sequential(
            nn.Linear(self.SEMANTIC_CONDITION_DIM + finger_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.SiLU(),
        )

    @staticmethod
    def build_relation_features(semantic_contact: torch.Tensor) -> torch.Tensor:
        """Keep semantic relations separate; AdaLN applies curriculum once as its mask."""
        curriculum = semantic_contact[:, :, CURRICULUM_WEIGHT]
        expected = semantic_contact[:, :, EXPECTED_CONTACT]
        return torch.stack(
            [
                expected,
                expected * semantic_contact[:, :, RED_PROXIMITY],
                semantic_contact[:, :, RED_CONTACT],
                semantic_contact[:, :, BLACK_PROXIMITY],
                semantic_contact[:, :, BLACK_VIOLATION],
                curriculum,
            ],
            dim=-1,
        )

    def forward(self, semantic_contact: torch.Tensor) -> torch.Tensor:
        if semantic_contact.ndim != 3:
            raise ValueError(
                "semantic_contact must have shape [batch, 5, feature_dim], "
                f"got {tuple(semantic_contact.shape)}"
            )
        if semantic_contact.shape[1] != self.NUM_CONTACT_LINKS:
            raise ValueError(
                f"semantic_contact must describe {self.NUM_CONTACT_LINKS} contact links, "
                f"got {semantic_contact.shape[1]}"
            )
        if semantic_contact.shape[2] <= CURRICULUM_WEIGHT:
            raise ValueError(
                "semantic_contact does not contain all required semantic channels"
            )

        relation_features = self.build_relation_features(semantic_contact)
        finger_ids = torch.arange(
            self.NUM_CONTACT_LINKS, device=semantic_contact.device
        )
        finger_embedding = self.finger_embedding(finger_ids).unsqueeze(0).expand(
            semantic_contact.shape[0], -1, -1
        )
        return self.net(torch.cat([relation_features, finger_embedding], dim=-1))


class SemanticAdaLNZero(nn.Module):
    """Baseline-preserving AdaLN-Zero modulation for pre-attention tokens."""

    def __init__(
        self,
        d_model: int,
        gamma_max: float = 0.1,
        beta_max: float = 0.1,
        gate_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.gamma_max = gamma_max
        self.beta_max = beta_max
        self.gate_max = gate_max
        self.modulation = nn.Linear(d_model, 2 * d_model + 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        normalized_tokens: torch.Tensor,
        semantic_condition: torch.Tensor,
        semantic_token_mask: torch.Tensor,
    ):
        modulation = self.modulation(semantic_condition)
        delta_gamma, beta, delta_gate = torch.split(
            modulation, [self.d_model, self.d_model, 1], dim=-1
        )
        delta_gamma = self.gamma_max * torch.tanh(delta_gamma)
        beta = self.beta_max * torch.tanh(beta)
        delta_gate = self.gate_max * torch.tanh(delta_gate)

        # Explicit masking is required: after training, the projection bias must
        # still be unable to modulate policy, palm, tool, object, or other links.
        delta_gamma = delta_gamma * semantic_token_mask
        beta = beta * semantic_token_mask
        delta_gate = delta_gate * semantic_token_mask
        conditioned_tokens = (1.0 + delta_gamma) * normalized_tokens + beta
        return conditioned_tokens, 1.0 + delta_gate


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        layer_norm_eps=1e-5,
        use_semantic_adaln=False,
    ):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.semantic_adaln = (
            SemanticAdaLNZero(d_model) if use_semantic_adaln else None
        )
        
        self.activation = getattr(F, activation)

    def forward(
        self,
        src,
        src_mask=None,
        semantic_condition=None,
        semantic_token_mask=None,
    ):
        # src: [Batch, SeqLen, D_model]
        # src_mask: [Batch * Heads, SeqLen, SeqLen] (The complex mask)
        
        # 1. Self Attention (Pre-Norm)
        src2 = self.norm1(src)
        residual_gate = None
        if self.semantic_adaln is not None and semantic_condition is not None:
            if semantic_token_mask is None:
                raise ValueError(
                    "semantic_token_mask is required when semantic_condition is provided"
                )
            src2, residual_gate = self.semantic_adaln(
                src2, semantic_condition, semantic_token_mask
            )
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, need_weights=False)
        if residual_gate is not None:
            src2 = residual_gate * src2
        src = src + self.dropout1(src2)
        
        # 2. Feed Forward (Pre-Norm)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class DexHandTopology(nn.Module):
    """URDF topology in the canonical rigid-body order used by observations."""

    def __init__(self, rh_spec=None, lh_spec=None):
        super().__init__()
        self.rh_spec = rh_spec or get_hand_spec("artimano", "right")
        self.lh_spec = lh_spec or get_hand_spec("artimano", "left")
        self.lh_offset = self.rh_spec.n_bodies
        self.tool_node_id = self.lh_offset + self.lh_spec.n_bodies
        self.object_node_id = self.tool_node_id + 1
        self.num_phys_nodes = self.object_node_id + 1
        self.register_buffer("dist_mat", self._build_dist_mat())
        self.register_buffer("serial_mask", self._build_serial_mask())
        self.register_buffer("synergy_mask", self._build_synergy_mask())
        self.register_buffer("centrality", self._build_centrality())

    def _build_dist_mat(self):
        dists = torch.full((self.num_phys_nodes, self.num_phys_nodes), 10, dtype=torch.long)
        dists.fill_diagonal_(0)
        for spec, offset in ((self.rh_spec, 0), (self.lh_spec, self.lh_offset)):
            for parent, child in spec.graph_edges:
                dists[parent + offset, child + offset] = 1
                dists[child + offset, parent + offset] = 1
            palm = spec.root_body_id + offset
            for node in (self.tool_node_id, self.object_node_id):
                dists[palm, node] = dists[node, palm] = 1
        dists[self.tool_node_id, self.object_node_id] = 1
        dists[self.object_node_id, self.tool_node_id] = 1
        for k in range(self.num_phys_nodes):
            dists = torch.minimum(dists, dists[:, k:k + 1] + dists[k:k + 1, :])
        return dists.clamp(max=5)

    def _build_serial_mask(self):
        return torch.where(self.dist_mat <= 1, 0.0, float("-inf"))

    def _build_synergy_mask(self):
        mask = torch.full((self.num_phys_nodes, self.num_phys_nodes), float("-inf"))
        mask.fill_diagonal_(0.0)
        for spec, offset in ((self.rh_spec, 0), (self.lh_spec, self.lh_offset)):
            for group in spec.synergy_groups:
                for i in group:
                    for j in group:
                        mask[i + offset, j + offset] = 0.0
        return mask

    def _build_centrality(self):
        # Preserve the Artimano checkpoint's palm/finger/object convention.
        degrees = torch.ones(self.num_phys_nodes, dtype=torch.long)
        for spec, offset in ((self.rh_spec, 0), (self.lh_spec, self.lh_offset)):
            degrees[spec.root_body_id + offset] = len(set(group[0] for group in spec.finger_groups))
        degrees[self.tool_node_id:] = 0
        return degrees


class ManoTopology(DexHandTopology):
    """Backward-compatible name for the default Artimano topology."""

class PhysicallyGroundedBias(nn.Module):
    # Semantic relations are intentionally excluded from attention logits in A1.
    # They condition RH contact-link tokens through AdaLN-Zero instead.
    TOOL_NODE_ID = 56
    OBJECT_NODE_ID = 57

    def __init__(self, topo: DexHandTopology, num_heads=8):
        super().__init__()
        self.topo = topo
        self.TOOL_NODE_ID = topo.tool_node_id
        self.OBJECT_NODE_ID = topo.object_node_id
        self.num_heads = num_heads
        self.spatial_bias = nn.Embedding(6, num_heads)
        self.edge_encoder = nn.Embedding(4, num_heads)
        nn.init.zeros_(self.spatial_bias.weight)
        nn.init.zeros_(self.edge_encoder.weight)
        # self.topo_bias_scale = nn.Parameter(torch.tensor(1e-3))
        self.topo_bias_scale = nn.Parameter(torch.tensor(0.0))
        # Near-zero rather than exact zero keeps the legacy bias initially
        # negligible while allowing the zero-initialized edge embeddings to learn.
        self.edge_bias_scale = nn.Parameter(torch.tensor(1e-3))
        self.geo_bias_scale = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.w_geo = nn.Linear(1, num_heads)

        max_deg = int(self.topo.centrality.max().item())
        self.cls_centrality_bias = nn.Embedding(max_deg + 1, num_heads)
        nn.init.zeros_(self.cls_centrality_bias.weight)
        self.cls_centrality_scale = nn.Parameter(torch.tensor(0.0))

        # Define Head Roles (Indices)
        # Assign 2 heads to Serial, 2 to Synergy, 4 to Global
        self.serial_heads = [0, 1] if num_heads >= 2 else []
        self.synergy_heads = [2, 3] if num_heads >= 4 else []
        # Heads 4,5,6,7 remain Global (unmasked)

        self.serial_scale = nn.Parameter(torch.tensor(0.0)) 
        self.synergy_scale = nn.Parameter(torch.tensor(0.0))

        # Retain the legacy state-dict key so semantic-bias checkpoints remain
        # loadable. This parameter is frozen and is not used by A1 attention.
        semantic_scale_init = torch.full((5, num_heads), 1e-3)
        semantic_scale_init[3:] = -1e-3
        self.semantic_relation_scale = nn.Parameter(
            semantic_scale_init, requires_grad=False
        )

        # Dataset/contact-body order: thumb, index, middle, ring, pinky.
        self.register_buffer(
            "rh_contact_node_ids",
            torch.tensor(topo.rh_spec.contact_body_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "lh_contact_node_ids",
            torch.tensor([i + topo.lh_offset for i in topo.lh_spec.contact_body_ids], dtype=torch.long),
            persistent=False,
        )
        unique_nodes = tuple(dict.fromkeys(topo.rh_spec.contact_body_ids))
        self.register_buffer("unique_rh_contact_node_ids", torch.tensor(unique_nodes), persistent=False)
        self.rh_contact_slot_groups = tuple(
            tuple(i for i, value in enumerate(topo.rh_spec.contact_body_ids) if value == node)
            for node in unique_nodes
        )

        # Restrict the generic geometric bias to the physically relevant,
        # side-specific contact pairs. The mask is symmetric because attention
        # is allowed in both query directions:
        #   RH contact links <-> tool, LH contact links <-> object.
        geometric_pair_mask = torch.zeros(
            self.topo.num_phys_nodes,
            self.topo.num_phys_nodes,
            dtype=torch.float32,
        )
        geometric_pair_mask[self.rh_contact_node_ids, self.TOOL_NODE_ID] = 1.0
        geometric_pair_mask[self.TOOL_NODE_ID, self.rh_contact_node_ids] = 1.0
        geometric_pair_mask[self.lh_contact_node_ids, self.OBJECT_NODE_ID] = 1.0
        geometric_pair_mask[self.OBJECT_NODE_ID, self.lh_contact_node_ids] = 1.0
        self.register_buffer(
            "geometric_pair_mask", geometric_pair_mask, persistent=False
        )

        # --- SPEED OPTIMIZATION: Pre-calculate indices ---
        # Instead of creating tensors in forward, create the index map once.
        N = self.topo.num_phys_nodes
        static_edge_indices = torch.zeros((N, N), dtype=torch.long)
        
        # 1. Kinematic Edges
        # Access the dist_mat from topo
        dist_cpu = self.topo._build_dist_mat() 
        static_edge_indices[dist_cpu == 1] = 1
        
        # 2. Self Edges
        static_edge_indices[range(N), range(N)] = 3
        self.register_buffer('static_edge_indices', static_edge_indices)

        static_edge_relations = torch.zeros((N, N, 4), dtype=torch.float32)
        static_edge_relations[..., 0][static_edge_indices == 0] = 1.0
        static_edge_relations[..., 1][dist_cpu == 1] = 1.0
        static_edge_relations[range(N), range(N), 3] = 1.0
        self.register_buffer("static_edge_relations", static_edge_relations, persistent=False)

    def forward(
        self,
        B,
        device,
        semantic_contact=None,
        x_pos=None,
        physical_contact=None,
    ):
        # 1. Topological Bias
        topo_bias = self.topo_bias_scale*self.spatial_bias(self.topo.dist_mat).permute(2,0,1).unsqueeze(0)

        # 2. Edge Bias
        static_edge_bias = torch.einsum(
            "ijr,rh->hij", self.static_edge_relations, self.edge_encoder.weight
        ).unsqueeze(0)
        soft_bias = topo_bias + self.edge_bias_scale * static_edge_bias
        nodes = self.unique_rh_contact_node_ids
        tool_node = self.TOOL_NODE_ID
        # Backward-compatible fallback for observations created before the
        # standalone physical_contact field was introduced.
        if physical_contact is None and semantic_contact is not None:
            physical_contact = semantic_contact[:, :, ACTUAL_CONTACT]
        if physical_contact is not None:
            # Allegro's ring/pinky semantic slots share a physical link. A link
            # gets one physical edge, with the strongest observation of contact.
            physical_contact = torch.stack([
                physical_contact[:, slots].amax(dim=-1)
                for slots in self.rh_contact_slot_groups
            ], dim=-1)
            # For a soft physical edge, interpolate only its legacy NONE/PHYSICAL
            # channel. This path is independent of semanticContact.enabled.
            physical_delta = physical_contact[:, None, :] * (
                self.edge_encoder.weight[2] - self.edge_encoder.weight[0]
            ).reshape(1, -1, 1)
            soft_bias = soft_bias.expand(B, -1, -1, -1).clone()
            soft_bias[:, :, nodes, tool_node] += self.edge_bias_scale * physical_delta
            soft_bias[:, :, tool_node, nodes] += self.edge_bias_scale * physical_delta
        else:
            soft_bias = soft_bias.expand(B, -1, -1, -1)

        # 3. Sparse geometric bias. A1 preserves its original all-head use.
        if x_pos is not None:
            dists = torch.cdist(x_pos, x_pos)
            sigma_sq = self.sigma.square().clamp_min(1e-8)
            geo_affinity = torch.exp(-dists.square() / (2.0 * sigma_sq))
            geo = self.w_geo(geo_affinity.unsqueeze(-1)).permute(0, 3, 1, 2)
            geo_pair_mask = self.geometric_pair_mask.unsqueeze(0).unsqueeze(0)
            soft_bias = soft_bias + self.geo_bias_scale * (geo * geo_pair_mask)
        
        
        # 4. Anatomical Masks
        # serial_mask in topo has 0.0 for valid and -inf for invalid.
        # 1.0 for valid, 0.0 for invalid.
        is_serial_valid = (self.topo.serial_mask == 0.0).float() 
        is_synergy_valid = (self.topo.synergy_mask == 0.0).float()
        # For Serial Heads (head 0, 1)
        for h in self.serial_heads:
            # Instead of forcing -inf, we ADD a bonus to the valid spots
            # If scale is positive, it boosts attention there.
            soft_bias[:, h] = soft_bias[:, h] + (self.serial_scale * is_serial_valid)
        # For Synergy Heads (heads 2, 3)
        for h in self.synergy_heads:
            soft_bias[:, h] = soft_bias[:, h] + (self.synergy_scale * is_synergy_valid)
        return soft_bias

class MLP_new(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLP_action(nn.Module):
    def __init__(self, action_dim=56):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class MLP_value(nn.Module):
    def __init__(self, privileged_dim=118):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(64 + privileged_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LATENT_ENCODER(nn.Module):
    """
    Transformer-based policy for bimanual hand–tool–object manipulation
    with per-link tokens, per-modality encoders, and structured target tokens.
    """

    # Type ids for type_embedding
    TYPE_LINK        = 0
    TYPE_HAND        = 1
    TYPE_TOOL        = 2
    TYPE_OBJECT      = 3
    TYPE_HAND_TARGET = 4  # RH_target, LH_target
    TYPE_POLICY      = 5

    # Side ids for side_embedding
    SIDE_NONE  = 0
    SIDE_RIGHT = 1
    SIDE_LEFT  = 2

    def __init__(
        self,
        rh_target_in_dim: int = 256,
        lh_target_in_dim: int = 256,
        tool_target_in_dim: int = 128,
        obj_target_in_dim: int = 128,
        action_dim: int = None,
        d_model: int = 64,
        num_heads: int = 8,
        num_global_layers: int = 4,
        mlp_hidden_dim: int = 64,
        dexhand: str = "artimano",
        obs_future_length: int = 1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_global_layers = num_global_layers
        self.rh_spec = get_hand_spec(dexhand, "right")
        self.lh_spec = get_hand_spec(dexhand, "left")
        self.layout = ObservationLayout(self.rh_spec, future_length=obs_future_length)
        action_dim = action_dim if action_dim is not None else 2 * (6 + self.rh_spec.n_dofs)
        self.action_dim = action_dim
        if d_model != 64:
            raise ValueError("The existing modality tokenizers require d_model=64")
        if action_dim != 2 * (6 + self.rh_spec.n_dofs):
            raise ValueError(f"action_dim={action_dim} does not match {dexhand} DOFs")

        # Retain even the unused legacy tokenizer modules and their names so
        # existing Artimano checkpoints still load strictly.
        link_flat_dim = (self.rh_spec.n_bodies - 1) * self.layout.link_token_dim
        self.rlink_tokenizer = MLP_new(link_flat_dim)
        self.llink_tokenizer = MLP_new(link_flat_dim)
        self.rh_tokenizer = MLP_new(self.layout.hand_token_dim)
        self.lh_tokenizer = MLP_new(self.layout.hand_token_dim)
        self.tool_tokenizer = MLP_new(129)
        self.obj_tokenizer = MLP_new(129)
        self.rh_target_tokenizer = MLP_new(76)
        self.lh_target_tokenizer = MLP_new(76)

        self.rh_curr_tokenizer = MLP_new(self.layout.curr_token_dim)
        self.rlink_curr_tokenizer = MLP_new(self.layout.link_token_dim)
        self.lh_curr_tokenizer = MLP_new(self.layout.curr_token_dim)
        self.llink_curr_tokenizer = MLP_new(self.layout.link_token_dim)
        self.palm_fuser = MLP_new(128)

        # Type & side embeddings
        self.type_embedding = nn.Embedding(6, d_model)
        self.side_embedding = nn.Embedding(3, d_model)
        self.id_embed = nn.Embedding(self.rh_spec.n_bodies + self.lh_spec.n_bodies + 2, d_model)

        # Learnable policy token
        self.policy_token_param = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.policy_token_param, std=0.02)


        self.rlink_tok_ln = nn.LayerNorm(64)
        self.llink_tok_ln = nn.LayerNorm(64)
        self.rh_tok_ln    = nn.LayerNorm(64)
        self.lh_tok_ln    = nn.LayerNorm(64)
        self.tool_tok_ln  = nn.LayerNorm(64)
        self.obj_tok_ln   = nn.LayerNorm(64)
        self.rh_tgt_ln    = nn.LayerNorm(64)
        self.lh_tgt_ln    = nn.LayerNorm(64)

        self.topo = DexHandTopology(self.rh_spec, self.lh_spec)
        self.bias_gen = PhysicallyGroundedBias(self.topo, num_heads)
        self.semantic_condition_encoder = SemanticConditionEncoder(d_model=d_model)

        physical_token_node_ids = torch.cat([
            torch.tensor([0, 0]),
            torch.arange(1, self.rh_spec.n_bodies),
            torch.tensor([self.topo.lh_offset, self.topo.lh_offset]),
            torch.arange(self.topo.lh_offset + 1, self.topo.tool_node_id),
            torch.tensor([self.topo.tool_node_id, self.topo.object_node_id]),
        ])
        self.register_buffer(
            "physical_token_node_ids", physical_token_node_ids, persistent=False
        )
        rh_contact_token_ids = torch.stack(
            [
                (physical_token_node_ids == node).nonzero(as_tuple=False)[0, 0] + 1
                for node in self.bias_gen.rh_contact_node_ids
            ]
        )
        self.register_buffer(
            "rh_contact_token_ids", rh_contact_token_ids, persistent=False
        )
        contact_counts = torch.zeros(len(physical_token_node_ids) + 1)
        contact_counts.index_add_(0, rh_contact_token_ids, torch.ones(5))
        self.register_buffer("rh_contact_token_counts", contact_counts.clamp_min(1), persistent=False)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dim_feedforward=512, 
                dropout=0.0,
                activation="relu",
                use_semantic_adaln=layer_idx >= max(0, num_global_layers - 2),
            )
            for layer_idx in range(num_global_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def build_semantic_token_condition(
        self,
        semantic_contact: torch.Tensor,
        tokens: torch.Tensor,
    ):
        """Scatter five per-finger conditions onto only the RH contact tokens."""
        if semantic_contact is None:
            return None, None

        finger_condition = self.semantic_condition_encoder(semantic_contact)
        finger_condition = finger_condition.to(dtype=tokens.dtype)
        token_condition = tokens.new_zeros(tokens.shape)
        token_condition = token_condition.index_add(
            1, self.rh_contact_token_ids, finger_condition
        )
        token_condition = token_condition / self.rh_contact_token_counts[None, :, None]
        token_mask = tokens.new_zeros(tokens.shape[0], tokens.shape[1], 1)
        curriculum_mask = semantic_contact[:, :, CURRICULUM_WEIGHT].unsqueeze(-1)
        curriculum_mask = curriculum_mask.to(dtype=tokens.dtype)
        token_mask = token_mask.index_add(
            1, self.rh_contact_token_ids, curriculum_mask
        )
        token_mask = token_mask / self.rh_contact_token_counts[None, :, None]
        return token_condition, token_mask

    # type+side embeddings and token builders
    def add_type_side_emb(self, x: torch.Tensor, type_id: int, side_id: int) -> torch.Tensor:
        dev = x.device
        type_tensor = torch.full((1,), type_id, dtype=torch.long, device=dev)
        side_tensor = torch.full((1,), side_id, dtype=torch.long, device=dev)
        type_emb = self.type_embedding(type_tensor)[0]  # [D]
        side_emb = self.side_embedding(side_tensor)[0]  # [D]
        return x + type_emb + side_emb

    def build_rlink_tokens(self, link_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.rlink_tok_ln(self.rlink_tokenizer(link_obs))
        return feat[:, None, :]

    def build_llink_tokens(self, link_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.llink_tok_ln(self.llink_tokenizer(link_obs))
        return feat[:, None, :]

    def build_rh_token(self, hand_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.rh_tok_ln(self.rh_tokenizer(hand_obs))
        return feat.unsqueeze(1)
    
    def build_lh_token(self, hand_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.lh_tok_ln(self.lh_tokenizer(hand_obs))
        return feat.unsqueeze(1)
    
    def build_rh_curr_token(self, hand_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.rh_tok_ln(self.rh_curr_tokenizer(hand_obs))
        return feat
    
    def build_lh_curr_token(self, hand_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.lh_tok_ln(self.lh_curr_tokenizer(hand_obs))
        return feat
    
    def build_rlink_curr_token(self, hand_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.rh_tok_ln(self.rlink_curr_tokenizer(hand_obs))
        return feat
    
    def build_llink_curr_token(self, hand_obs: Dict[str, torch.Tensor], side_id: int) -> torch.Tensor:
        feat = self.lh_tok_ln(self.llink_curr_tokenizer(hand_obs))
        return feat

    def build_tool_token(self, tool_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.tool_tok_ln(self.tool_tokenizer(tool_obs))
        return feat.unsqueeze(1)

    def build_object_token(self, obj_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.obj_tok_ln(self.obj_tokenizer(obj_obs))
        return feat.unsqueeze(1)

    def build_rh_target_token(self, rh_target_obs: torch.Tensor) -> torch.Tensor:
        feat = self.rh_tgt_ln(self.rh_target_tokenizer(rh_target_obs))
        return feat.unsqueeze(1)

    def build_lh_target_token(self, lh_target_obs: torch.Tensor) -> torch.Tensor:
        feat = self.lh_tgt_ln(self.lh_target_tokenizer(lh_target_obs))
        return feat.unsqueeze(1)

    def build_history_token(self, prev_action: torch.Tensor) -> torch.Tensor:
        feat = self.history_encoder(prev_action)
        return feat.unsqueeze(1)

    def build_policy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        policy_vec = self.policy_token_param.view(1, 1, -1).expand(batch_size, 1, -1)
        return policy_vec

    def unpack_hand_observation(self, prop, priv, target):
        """Read field-major environment targets without mixing future frames."""
        fields = {name: target[:, index] for name, index in self.layout.target_slices.items()}
        hand_obs = torch.cat([prop, priv[:, :-1]], dim=-1)
        current_fields = (
            "delta_wrist_pos", "wrist_vel", "delta_wrist_vel", "wrist_quat",
            "delta_wrist_quat", "wrist_ang_vel", "delta_wrist_ang_vel",
            "delta_manip_obj_pos", "manip_obj_vel", "delta_manip_obj_vel",
            "manip_obj_quat", "delta_manip_obj_quat", "manip_obj_ang_vel",
            "delta_manip_obj_ang_vel", "obj_to_joints",
        )
        current_obs = torch.cat([fields[name] for name in current_fields], dim=-1).unsqueeze(1)
        batch_size = target.shape[0]
        link_fields = [
            fields[name].reshape(batch_size, self.layout.future_length, self.rh_spec.n_bodies - 1, 3)
            for name in ("delta_joints_pos", "joints_vel", "delta_joints_vel")
        ]
        link_obs = torch.stack(link_fields, dim=3).permute(0, 2, 1, 3, 4)
        link_obs = link_obs.reshape(batch_size, self.rh_spec.n_bodies - 1, self.layout.link_token_dim)
        object_obs = torch.cat([priv[:, -1:], fields["bps"]], dim=-1)
        return hand_obs, current_obs, link_obs, object_obs

    def forward(self, obs: Dict[str, Any], pre_action=None) -> Dict[str, torch.Tensor]:
        dimensions = {
            "proprioception": 2 * self.layout.hand_proprioception_dim,
            "privileged": 2 * self.layout.hand_privileged_dim,
            "target": 2 * self.layout.hand_target_dim,
        }
        for name, width in dimensions.items():
            if obs[name].ndim != 2 or obs[name].shape[1] != width:
                raise ValueError(f"{name} must have shape [batch, {width}], got {tuple(obs[name].shape)}")
        if obs["phys_node_pos"].shape[1:] != (self.topo.num_phys_nodes, 3):
            raise ValueError(f"phys_node_pos must describe {self.topo.num_phys_nodes} graph nodes")
        r_prop, l_prop = obs["proprioception"].split(self.layout.hand_proprioception_dim, dim=-1)
        r_priv, l_priv = obs["privileged"].split(self.layout.hand_privileged_dim, dim=-1)
        r_target, l_target = obs["target"].split(self.layout.hand_target_dim, dim=-1)
        rh_obs, rh_curr_obs, rh_links_obs_curr, tool_obs = self.unpack_hand_observation(r_prop, r_priv, r_target)
        lh_obs, lh_curr_obs, lh_links_obs_curr, obj_obs = self.unpack_hand_observation(l_prop, l_priv, l_target)

        device = tool_obs.device
        B = tool_obs.shape[0]

        # 1) Build tokens
        rh_hand_tok = self.build_rh_token(rh_obs, side_id=self.SIDE_RIGHT)
        lh_hand_tok = self.build_lh_token(lh_obs, side_id=self.SIDE_LEFT)

        tool_tok = self.build_tool_token(tool_obs)
        obj_tok  = self.build_object_token(obj_obs)

        rh_curr_tok = self.build_rh_curr_token(rh_curr_obs, side_id=self.SIDE_RIGHT)
        lh_curr_tok = self.build_lh_curr_token(lh_curr_obs, side_id=self.SIDE_LEFT)
        rh_links_curr_tok = self.build_rlink_curr_token(rh_links_obs_curr, side_id=self.SIDE_RIGHT)
        lh_links_curr_tok = self.build_llink_curr_token(lh_links_obs_curr, side_id=self.SIDE_LEFT)

        policy_tok  = self.build_policy_token(B, device)

        # 5) Stage 3: Global Transformer
        tokens = torch.cat(
            [
                rh_hand_tok,
                rh_curr_tok,
                rh_links_curr_tok,
                lh_hand_tok,
                lh_curr_tok,
                lh_links_curr_tok,
                tool_tok,
                obj_tok,
            ],
            dim=1,
        ) 

        tokens = torch.cat(
            [
                policy_tok,
                tokens
            ],
            dim=1,
        )
        
        semantic_contact = obs.get("semantic_contact")
        physical_contact = obs.get("physical_contact")
        x_pos = obs["phys_node_pos"]
        attn_bias_base = self.bias_gen(
            B,
            device,
            semantic_contact=semantic_contact,
            physical_contact=physical_contact,
            x_pos=x_pos,
        )
        # Each palm has two tokens sharing one physical graph node.
        # sequence: [RH_Palm, RH_Curr, RH_Links..., LH_Palm, LH_Curr, LH_Links..., Tool, Obj]
        idx_map = self.physical_token_node_ids
        # Expand the graph bias to tokens, then pad the policy-token row/column.
        attn_bias = attn_bias_base[:, :, idx_map, :][:, :, :, idx_map]
        attn_bias = F.pad(attn_bias, (1,0, 1,0), value=0.0).to(device)

        deg_base = self.topo.centrality.to(device)
        deg = deg_base[idx_map]
        deg = deg.clamp(min=0, max=self.bias_gen.cls_centrality_bias.num_embeddings - 1)
        b = self.bias_gen.cls_centrality_bias(deg) 
        b = b.t().unsqueeze(0) 
        attn_bias[:, :, 0, 1:] += self.bias_gen.cls_centrality_scale * b
        attn_bias[:, :, 1:, 0] += self.bias_gen.cls_centrality_scale * b

        semantic_condition, semantic_token_mask = self.build_semantic_token_condition(
            semantic_contact, tokens
        )
        for layer in self.layers:
            tokens = layer(
                tokens,
                src_mask=attn_bias,
                semantic_condition=semantic_condition,
                semantic_token_mask=semantic_token_mask,
            )
        tokens = self.final_norm(tokens)

        # 6) Read policy token -> action/value
        policy_out = tokens[:, 0, :]  # [B,D]
        return policy_out

        
class BimanualHandToolPolicy(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        NetworkBuilder.BaseNetwork.__init__(self)
        params = params or {}
        if params.get("policy_mode", "direct") != "direct":
            raise ValueError("BimanualHandToolPolicy supports policy_mode='direct' only")
        self.dexhand = str(params.get("dexhand", "artimano")).lower()
        self.rh_spec = get_hand_spec(self.dexhand, "right")
        self.lh_spec = get_hand_spec(self.dexhand, "left")
        self.obs_future_length = params.get("obs_future_length", 1)
        self.layout = ObservationLayout(self.rh_spec, future_length=self.obs_future_length)
        action_dim = int(kwargs.get("actions_num", 2 * (6 + self.rh_spec.n_dofs)))
        expected_action_dim = 2 * (6 + self.rh_spec.n_dofs)
        if action_dim != expected_action_dim:
            raise ValueError(
                f"{self.dexhand} direct policy requires {expected_action_dim} actions, got {action_dim}"
            )
        self.tau = 0.01
        self.latent_encoder = LATENT_ENCODER(
            dexhand=self.dexhand,
            obs_future_length=self.obs_future_length,
            action_dim=action_dim,
        )
        if params.get("compile_encoder", True):
            try:
                self.latent_encoder = torch.compile(self.latent_encoder)
            except Exception as exc:
                print(f"Couldn't compile encoder: {exc}")
        self.action_miu_head = MLP_action(action_dim)
        self.value_head = MLP_value(2 * self.layout.hand_privileged_dim)
        self.sigma = nn.Parameter(
            -torch.ones(action_dim, dtype=torch.float), requires_grad=True
        )
        self._checkpoint_metadata = {
            "version": 1,
            "policy_mode": "direct",
            "rh_hand": self.rh_spec.signature(),
            "lh_hand": self.lh_spec.signature(),
            "observation_layout_version": 1,
            "obs_future_length": self.obs_future_length,
            "action_dim": action_dim,
        }

    def get_extra_state(self):
        return deepcopy(self._checkpoint_metadata)

    def set_extra_state(self, state):
        if state != self._checkpoint_metadata:
            raise RuntimeError(
                "Checkpoint hand/DOF/body ordering or observation layout does not match "
                f"the current {self.dexhand} direct policy"
            )

    def prepare_checkpoint_state_dict(self, state_dict, prefix=""):
        """Validate embodiment before copying weights; accept old Artimano models."""
        metadata_key = prefix + "_extra_state"
        if metadata_key not in state_dict:
            if self.dexhand != "artimano" or self.obs_future_length != 1:
                raise RuntimeError(
                    "Checkpoint has no hand metadata. Only legacy Artimano "
                    "checkpoints with obs_future_length=1 can be loaded without metadata."
                )
            state_dict[metadata_key] = self.get_extra_state()
        self.set_extra_state(state_dict[metadata_key])

        # torch.compile adds _orig_mod to state-dict paths. Permit restoring a
        # checkpoint with compilation either enabled or disabled.
        source_prefix = prefix + "latent_encoder."
        compiled = hasattr(self.latent_encoder, "_orig_mod")
        for key in tuple(state_dict):
            if not key.startswith(source_prefix):
                continue
            suffix = key[len(source_prefix):]
            if suffix.startswith("_orig_mod."):
                suffix = suffix[len("_orig_mod."):]
            destination = source_prefix + ("_orig_mod." if compiled else "") + suffix
            if destination != key:
                state_dict[destination] = state_dict.pop(key)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.prepare_checkpoint_state_dict(state_dict, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, obs: Dict[str, Any], pre_action=None) -> Dict[str, torch.Tensor]:
        z = self.latent_encoder(obs, pre_action)
        action_miu = self.action_miu_head(z)
        value = self.value_head(torch.cat([z, obs["privileged"]], dim=-1))
        return action_miu, self.sigma, value, None, None
    
class ResBiHDictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        net = BimanualHandToolPolicy(self.params, **kwargs)
        return net

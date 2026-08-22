import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
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


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu", layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None):
        # src: [Batch, SeqLen, D_model]
        # src_mask: [Batch * Heads, SeqLen, SeqLen] (The complex mask)
        
        # 1. Self Attention (Pre-Norm)
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2)
        
        # 2. Feed Forward (Pre-Norm)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class ManoTopology(nn.Module):
    """
    Exact 28-node graph for MANO.
    0: Palm, 1-5: Index, 6-10: Middle, 11-15: Pinky, 16-20: Ring, 21-27: Thumb
    """
    def __init__(self):
        super().__init__()
        self.num_phys_nodes = 58 
        self.register_buffer('dist_mat', self._build_dist_mat())
        self.register_buffer('serial_mask', self._build_serial_mask())
        self.register_buffer('synergy_mask', self._build_synergy_mask())
        self.register_buffer('centrality', self._build_centrality())

    def _build_dist_mat(self):
        dists = torch.full((self.num_phys_nodes, self.num_phys_nodes), 10, dtype=torch.long)#.cuda()
        dists.fill_diagonal_(0)
        def connect(i, j): dists[i, j] = 1; dists[j, i] = 1

        def wire_hand(offset):
            palm = offset
            # Index (1-5)
            connect(palm, offset+1); connect(offset+1, offset+2); connect(offset+2, offset+3)
            connect(offset+3, offset+4); connect(offset+4, offset+5)
            # Middle (6-10)
            connect(palm, offset+6); connect(offset+6, offset+7); connect(offset+7, offset+8)
            connect(offset+8, offset+9); connect(offset+9, offset+10)
            # Pinky (11-15)
            connect(palm, offset+11); connect(offset+11, offset+12); connect(offset+12, offset+13)
            connect(offset+13, offset+14); connect(offset+14, offset+15)
            # Ring (16-20)
            connect(palm, offset+16); connect(offset+16, offset+17); connect(offset+17, offset+18)
            connect(offset+18, offset+19); connect(offset+19, offset+20)
            # Thumb (21-27)
            connect(palm, offset+21); connect(offset+21, offset+22); connect(offset+22, offset+23)
            connect(offset+23, offset+24); connect(offset+24, offset+25); connect(offset+25, offset+26)
            connect(offset+26, offset+27)

        wire_hand(0); wire_hand(28)
        tool, obj = 56, 57
        # connect to palms
        connect(0, tool); connect(28, tool)
        connect(0, obj);  connect(28, obj)
        # tool-object edge
        connect(tool, obj)
        # Calculating Shortest Paths (Floyd-Warshall)
        for k in range(self.num_phys_nodes):
            for i in range(self.num_phys_nodes):
                for j in range(self.num_phys_nodes):
                    dists[i,j] = min(dists[i,j], dists[i,k] + dists[k,j])
        return dists.clamp(max=5)

    def _build_serial_mask(self):
        mask = torch.full((self.num_phys_nodes, self.num_phys_nodes), float('-inf'))#.cuda()
        mask[self._build_dist_mat() <= 1] = 0.0
        return mask

    def _build_synergy_mask(self):
        mask = torch.full((self.num_phys_nodes, self.num_phys_nodes), float('-inf'))#.cuda()
        mask.fill_diagonal_(0.0)
        tips = [5, 10, 15, 20, 27]
        mcps = [1, 6, 11, 16, 21]
        pips = [3, 8, 13, 18, 24]
        for grp in [tips, mcps, pips]:
            for x in [0, 28]: # RH then LH
                indices = [i + x for i in grp]
                for i in indices:
                    for j in indices: mask[i, j] = 0.0
        return mask

    def _build_centrality(self):
        degs = torch.ones(self.num_phys_nodes, dtype=torch.long)#.cuda()
        degs[0] = 5; degs[28] = 5; degs[56] = 0; degs[57] = 0
        return degs

class PhysicallyGroundedBias(nn.Module):
    # Semantic relation channels. Original topology/anatomy/geometry and the
    # legacy 4-way edge embedding remain separate additive sources.
    EXPECTED_REL = 0
    RED_PROXIMITY_REL = 1
    RED_CONTACT_REL = 2
    BLACK_PROXIMITY_REL = 3
    BLACK_VIOLATION_REL = 4

    def __init__(self, topo: ManoTopology, num_heads=8):
        super().__init__()
        self.topo = topo
        self.num_heads = num_heads
        self.spatial_bias = nn.Embedding(6, num_heads)
        self.edge_encoder = nn.Embedding(4, num_heads)
        nn.init.zeros_(self.spatial_bias.weight)
        nn.init.zeros_(self.edge_encoder.weight)
        self.topo_bias_scale = nn.Parameter(torch.tensor(1e-3))
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

        # Five new relation types are multi-hot and learnable. Near-zero scales
        # preserve the old Stage-1 behavior while avoiding a dead semantic path.
        # Start extremely close to the legacy policy while keeping the semantic
        # path non-zero and trainable from the first optimizer step.
        semantic_scale_init = torch.full((5, num_heads), 1e-3)
        semantic_scale_init[self.BLACK_PROXIMITY_REL :] = -1e-3
        self.semantic_relation_scale = nn.Parameter(semantic_scale_init)
        semantic_head_mask = torch.zeros(5, num_heads)
        red_heads = [h for h in (4, 5) if h < num_heads]
        black_heads = [h for h in (6, 7) if h < num_heads]
        semantic_head_mask[self.EXPECTED_REL, red_heads] = 1.0
        semantic_head_mask[self.RED_PROXIMITY_REL, red_heads] = 1.0
        semantic_head_mask[self.RED_CONTACT_REL, red_heads] = 1.0
        semantic_head_mask[self.BLACK_PROXIMITY_REL, black_heads] = 1.0
        semantic_head_mask[self.BLACK_VIOLATION_REL, black_heads] = 1.0
        self.register_buffer("semantic_head_mask", semantic_head_mask, persistent=False)

        # Dataset/contact-body order: thumb, index, middle, ring, pinky.
        self.register_buffer(
            "rh_contact_node_ids",
            torch.tensor([26, 4, 9, 19, 14], dtype=torch.long),
            persistent=False,
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

    def _semantic_relation_values(self, semantic_strengths):
        relation_weights = 2.0 * torch.tanh(self.semantic_relation_scale) * self.semantic_head_mask
        raw_bias = torch.einsum("bir,rh->bhi", semantic_strengths, relation_weights)
        # Several relations on one node pair remain present and additive, while
        # their combined semantic contribution is bounded to [-2, 2].
        return 2.0 * torch.tanh(raw_bias / 2.0)

    def _semantic_strengths(self, semantic_contact):
        curriculum = semantic_contact[:, :, CURRICULUM_WEIGHT]
        return torch.stack(
            [
                curriculum * semantic_contact[:, :, EXPECTED_CONTACT],
                curriculum * semantic_contact[:, :, EXPECTED_CONTACT]
                * semantic_contact[:, :, RED_PROXIMITY],
                curriculum * semantic_contact[:, :, RED_CONTACT],
                curriculum * semantic_contact[:, :, BLACK_PROXIMITY],
                curriculum * semantic_contact[:, :, BLACK_VIOLATION],
            ],
            dim=-1,
        )

    def forward(self, B, device, semantic_contact=None, x_pos=None):
        # 1. Topological Bias
        topo_bias = self.topo_bias_scale*self.spatial_bias(self.topo.dist_mat).permute(2,0,1).unsqueeze(0)

        # 2. Edge Bias
        static_edge_bias = torch.einsum(
            "ijr,rh->hij", self.static_edge_relations, self.edge_encoder.weight
        ).unsqueeze(0)
        soft_bias = topo_bias + self.edge_bias_scale * static_edge_bias
        if semantic_contact is not None:
            contact = semantic_contact[:, :, ACTUAL_CONTACT]
            nodes = self.rh_contact_node_ids
            tool_node = 56
            # For a soft physical edge, interpolate only its legacy NONE/PHYSICAL
            # channel while all other relations remain additive and untouched.
            physical_delta = contact[:, None, :] * (
                self.edge_encoder.weight[2] - self.edge_encoder.weight[0]
            ).reshape(1, -1, 1)
            soft_bias = soft_bias.expand(B, -1, -1, -1).clone()
            soft_bias[:, :, nodes, tool_node] += self.edge_bias_scale * physical_delta
            soft_bias[:, :, tool_node, nodes] += self.edge_bias_scale * physical_delta

            strengths = self._semantic_strengths(semantic_contact)
            semantic_values = self._semantic_relation_values(strengths)
            soft_bias[:, :, nodes, tool_node] += semantic_values
            soft_bias[:, :, tool_node, nodes] += semantic_values
        else:
            soft_bias = soft_bias.expand(B, -1, -1, -1)

        # 3. Geometric Bias
        if x_pos is not None:
            dists = torch.cdist(x_pos, x_pos)
            geo = self.w_geo(torch.exp(-dists**2/(2*self.sigma**2)).unsqueeze(-1)).permute(0,3,1,2)
            far_mask = (self.topo.dist_mat > 2).float().unsqueeze(0).unsqueeze(0)
            soft_bias = soft_bias + self.geo_bias_scale*(geo * far_mask)
        
        
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

    def add_policy_semantic_bias(self, attn_bias, semantic_contact, contact_token_ids):
        """Add the same semantic relations between contact links and policy token."""
        if semantic_contact is None:
            return attn_bias
        strengths = self._semantic_strengths(semantic_contact)
        values = self._semantic_relation_values(strengths)
        attn_bias[:, :, contact_token_ids, 0] += values
        attn_bias[:, :, 0, contact_token_ids] += values
        return attn_bias

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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 56),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class MLP_value(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(64+118, 256),
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
        action_dim: int = 56, #28 per hand
        d_model: int = 64,
        num_heads: int = 8,
        num_global_layers: int = 4,
        mlp_hidden_dim: int = 64
    ):
        super().__init__()

        self.d_model = d_model
        self.action_dim = action_dim

        self.rlink_tokenizer = MLP_new(243)
        self.llink_tokenizer = MLP_new(243)
        self.rh_tokenizer = MLP_new(137)
        self.lh_tokenizer = MLP_new(137)
        self.tool_tokenizer = MLP_new(129)
        self.obj_tokenizer = MLP_new(129)
        self.rh_target_tokenizer = MLP_new(76)
        self.lh_target_tokenizer = MLP_new(76)

        self.rh_curr_tokenizer = MLP_new(74)
        self.rlink_curr_tokenizer = MLP_new(9)
        self.lh_curr_tokenizer = MLP_new(74)
        self.llink_curr_tokenizer = MLP_new(9)
        self.palm_fuser = MLP_new(128)

        # Type & side embeddings
        self.type_embedding = nn.Embedding(6, d_model)
        self.side_embedding = nn.Embedding(3, d_model)
        self.id_embed = nn.Embedding(58, d_model)

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

        self.topo = ManoTopology()
        self.bias_gen = PhysicallyGroundedBias(self.topo, num_heads)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dim_feedforward=512, 
                dropout=0.0,
                activation="relu"
            )
            for _ in range(4) # num_global_layers
        ])
        self.final_norm = nn.LayerNorm(d_model)

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

    def forward(self, obs: Dict[str, Any], pre_action=None) -> Dict[str, torch.Tensor]:
        r_prop = obs['proprioception'][:, :79]
        l_prop = obs['proprioception'][:, 79:]
        r_priv = obs['privileged'][:, :59]
        l_priv = obs['privileged'][:, 59:]
        r_curr = obs['target'][:, None, :450]
        l_curr = obs['target'][:, None, 450:]

        r_q = r_prop[:, :22]
        r_cosq = r_prop[:, 22:44]
        r_sinq = r_prop[:, 44:66]
        r_base_state = r_prop[::, 66:]
        l_q = l_prop[:, :22]
        l_cosq = l_prop[:, 22:44]
        l_sinq = l_prop[:, 44:66]
        l_base_state = l_prop[::, 66:]

        r_dq = r_priv[:, :22]
        r_manip_obj_pos = r_priv[:, 22:25]
        r_manip_obj_quat = r_priv[:, 25:29]
        r_manip_obj_vel = r_priv[:, 29:32]
        r_manip_obj_ang_vel = r_priv[:, 32:35]
        r_tip_force = r_priv[:, 35:55]
        r_manip_obj_com = r_priv[:, 55:58]
        r_manip_obj_weight = r_priv[:, -1:]
        l_dq = l_priv[:, :22]
        l_manip_obj_pos = l_priv[:, 22:25]
        l_manip_obj_quat = l_priv[:, 25:29]
        l_manip_obj_vel = l_priv[:, 29:32]
        l_manip_obj_ang_vel = l_priv[:, 32:35]
        l_tip_force = l_priv[:, 35:55]
        l_manip_obj_com = l_priv[:, 55:58]
        l_manip_obj_weight = l_priv[:, -1:]

        r_delta_wrist_pos_curr = r_curr[:, :, :3]
        r_wrist_vel_curr = r_curr[:, :, 3:6]
        r_delta_wrist_vel_curr = r_curr[:, :, 6:9]
        r_wrist_quat_curr = r_curr[:, :, 9:13] 
        r_delta_wrist_quat_curr = r_curr[:, :, 13:17]
        r_wrist_ang_vel_curr = r_curr[:, :, 17:20] 
        r_delta_wrist_ang_vel_curr = r_curr[:, :, 20:23]
        r_delta_joints_pos_curr = r_curr[:, :, 23:104]
        r_joints_vel_target_curr = r_curr[:, :, 104:185] 
        r_delta_joints_vel_curr = r_curr[:, :, 185:266]
        r_delta_manip_obj_pos_curr = r_curr[:, :, 266:269]
        r_manip_obj_vel_target_curr = r_curr[:, :, 269:272]
        r_delta_manip_obj_vel_curr = r_curr[:, :, 272:275]
        r_manip_obj_quat_target_curr = r_curr[:, :, 275:279] 
        r_delta_manip_obj_quat_curr = r_curr[:, :, 279:283]
        r_manip_obj_ang_vel_target_curr = r_curr[:, :, 283:286] 
        r_delta_manip_obj_ang_vel_curr = r_curr[:, :, 286:289]
        r_obj_to_joints_curr = r_curr[:, :, 289:317]
        r_bps = r_curr[:, 0, 322:]
        l_delta_wrist_pos_curr = l_curr[:, :, :3]
        l_wrist_vel_curr = l_curr[:, :, 3:6]
        l_delta_wrist_vel_curr = l_curr[:, :, 6:9]
        l_wrist_quat_curr = l_curr[:, :, 9:13] 
        l_delta_wrist_quat_curr = l_curr[:, :, 13:17]
        l_wrist_ang_vel_curr = l_curr[:, :, 17:20] 
        l_delta_wrist_ang_vel_curr = l_curr[:, :, 20:23]
        l_delta_joints_pos_curr = l_curr[:, :, 23:104]
        l_joints_vel_target_curr = l_curr[:, :, 104:185] 
        l_delta_joints_vel_curr = l_curr[:, :, 185:266]
        l_delta_manip_obj_pos_curr = l_curr[:, :, 266:269]
        l_manip_obj_vel_target_curr = l_curr[:, :, 269:272]
        l_delta_manip_obj_vel_curr = l_curr[:, :, 272:275]
        l_manip_obj_quat_target_curr = l_curr[:, :, 275:279] 
        l_delta_manip_obj_quat_curr = l_curr[:, :, 279:283]
        l_manip_obj_ang_vel_target_curr = l_curr[:, :, 283:286] 
        l_delta_manip_obj_ang_vel_curr = l_curr[:, :, 286:289]
        l_obj_to_joints_curr = l_curr[:, :, 289:317]
        l_bps = l_curr[:, 0, 322:]

        rh_obs = torch.cat([r_q, r_cosq, r_sinq, r_base_state, r_dq, r_manip_obj_pos, r_manip_obj_quat,
                            r_manip_obj_vel, r_manip_obj_ang_vel, r_tip_force, r_manip_obj_com], dim=-1)
        lh_obs = torch.cat([l_q, l_cosq, l_sinq, l_base_state, l_dq, l_manip_obj_pos, l_manip_obj_quat,
                            l_manip_obj_vel, l_manip_obj_ang_vel, l_tip_force, l_manip_obj_com], dim=-1)
        rh_links_obs_curr = torch.cat([r_delta_joints_pos_curr, r_joints_vel_target_curr, r_delta_joints_vel_curr], dim=-1)
        lh_links_obs_curr = torch.cat([l_delta_joints_pos_curr, l_joints_vel_target_curr, l_delta_joints_vel_curr], dim=-1)
        rh_links_obs_curr = rh_links_obs_curr.view(rh_links_obs_curr.shape[0], 3, 27, 3)
        rh_links_obs_curr = torch.stack([rh_links_obs_curr[:, 0], rh_links_obs_curr[:, 1], rh_links_obs_curr[:, 2],], dim=2)
        rh_links_obs_curr = rh_links_obs_curr.view(rh_links_obs_curr.shape[0], rh_links_obs_curr.shape[1], -1)
        lh_links_obs_curr = lh_links_obs_curr.view(lh_links_obs_curr.shape[0], 3, 27, 3)
        lh_links_obs_curr = torch.stack([lh_links_obs_curr[:, 0], lh_links_obs_curr[:, 1], lh_links_obs_curr[:, 2],], dim=2)
        lh_links_obs_curr = lh_links_obs_curr.view(lh_links_obs_curr.shape[0], lh_links_obs_curr.shape[1], -1)
        rh_curr_obs = torch.cat([r_delta_wrist_pos_curr, r_wrist_vel_curr, r_delta_wrist_vel_curr, r_wrist_quat_curr, 
                                   r_delta_wrist_quat_curr, r_wrist_ang_vel_curr, r_delta_wrist_ang_vel_curr,
                                    r_delta_manip_obj_pos_curr, r_manip_obj_vel_target_curr, r_delta_manip_obj_vel_curr,
                                    r_manip_obj_quat_target_curr, r_delta_manip_obj_quat_curr, r_manip_obj_ang_vel_target_curr,
                                    r_delta_manip_obj_ang_vel_curr, r_obj_to_joints_curr], dim=-1)
        lh_curr_obs = torch.cat([l_delta_wrist_pos_curr, l_wrist_vel_curr, l_delta_wrist_vel_curr, l_wrist_quat_curr, 
                                   l_delta_wrist_quat_curr, l_wrist_ang_vel_curr, l_delta_wrist_ang_vel_curr,
                                    l_delta_manip_obj_pos_curr, l_manip_obj_vel_target_curr, l_delta_manip_obj_vel_curr,
                                    l_manip_obj_quat_target_curr, l_delta_manip_obj_quat_curr, l_manip_obj_ang_vel_target_curr,
                                    l_delta_manip_obj_ang_vel_curr, l_obj_to_joints_curr], dim=-1)
        tool_obs = torch.cat([r_manip_obj_weight, r_bps], dim=-1)
        obj_obs = torch.cat([l_manip_obj_weight, l_bps], dim=-1)


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
        x_pos = obs["phys_node_pos"]
        attn_bias_base = self.bias_gen(
            B,
            device,
            semantic_contact=semantic_contact,
            x_pos=x_pos,
        )
        # define mapping: 60 Physical Tokens -> 58 Graph Nodes
        # we duplicate Node 0 (RH Palm) and Node 28 (LH Palm)
        # sequence: [RH_Palm, RH_Curr, RH_Links..., LH_Palm, LH_Curr, LH_Links..., Tool, Obj]
        idx_map = torch.cat([
            torch.tensor([0, 0], device=device),           # RH Palm & Curr -> Node 0
            torch.arange(1, 28, device=device),            # RH Links
            torch.tensor([28, 28], device=device),         # LH Palm & Curr -> Node 28
            torch.arange(29, 56, device=device),           # LH Links
            torch.tensor([56, 57], device=device)          # Tool, Obj
        ]) # Shape: [60]
        # 3. Expand the Bias Matrix [B, H, 60, 60]
        # duplicate the rows/cols for the palms/wrists so they share the same bias
        attn_bias = attn_bias_base[:, :, idx_map, :][:, :, :, idx_map]
        attn_bias = F.pad(attn_bias, (1,0, 1,0), value=0.0).to(device)

        # Locate each RH contact graph node in the actual token sequence. The
        # +1 accounts for the policy token prepended above.
        contact_token_ids = torch.stack(
            [(idx_map == node).nonzero(as_tuple=False)[0, 0] + 1 for node in self.bias_gen.rh_contact_node_ids]
        )
        # semantic - policy
        attn_bias = self.bias_gen.add_policy_semantic_bias(
            attn_bias, semantic_contact, contact_token_ids
        )

        deg_base = self.topo.centrality.to(device)
        # Expand degrees to match the 60 tokens
        deg = deg_base[idx_map] # [60]
        deg = deg.clamp(min=0, max=self.bias_gen.cls_centrality_bias.num_embeddings - 1)
        b = self.bias_gen.cls_centrality_bias(deg) 
        b = b.t().unsqueeze(0) 
        attn_bias[:, :, 0, 1:] += self.bias_gen.cls_centrality_scale * b
        attn_bias[:, :, 1:, 0] += self.bias_gen.cls_centrality_scale * b

        for layer in self.layers:
            tokens = layer(tokens, src_mask=attn_bias)
        tokens = self.final_norm(tokens)

        # 6) Read policy token -> action/value
        policy_out = tokens[:, 0, :]  # [B,D]
        return policy_out

        
class BimanualHandToolPolicy(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        #super().__init__()
        NetworkBuilder.BaseNetwork.__init__(self)
        self.tau = 0.01
        self.latent_encoder = LATENT_ENCODER()
        try:
            self.latent_encoder = torch.compile(self.latent_encoder)
            print("Successfully compile LATENT_ENCODER with torch.compile hahaha")
        except Exception as e:
            print(f"Couldn't compile encoder: {e}")
        self.action_miu_head = MLP_action()
        self.value_head = MLP_value()
        action_dim = 56
        self.sigma = nn.Parameter(
            -torch.ones(action_dim, dtype=torch.float), requires_grad=True
        )

    def forward(self, obs: Dict[str, Any], pre_action=None) -> Dict[str, torch.Tensor]:
        z = self.latent_encoder(obs, pre_action)
        action_miu = self.action_miu_head(z)
        value = self.value_head(torch.cat([z, obs["privileged"]], dim=-1))
        return action_miu, self.sigma, value, None, None
    
class ResBiHDictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        net = BimanualHandToolPolicy(self.params, **kwargs)
        return net

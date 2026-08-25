"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
import torch

def edges_to_coo(edge_list, num_nodes=None):
    # Convert input to a LongTensor of shape [E, 2].
    if isinstance(edge_list, torch.Tensor):
        if edge_list.dtype != torch.long:
            edge_list = edge_list.to(dtype=torch.long)

        if edge_list.numel() == 0:
            edge_list = edge_list.reshape(0, 2)
        elif edge_list.dim() != 2 or edge_list.shape[1] != 2:
            raise ValueError("edge_list must have shape [E, 2].")
    else:
        if len(edge_list) == 0:
            edge_list = torch.empty((0, 2), dtype=torch.long)
        else:
            edge_list = torch.tensor(edge_list, dtype=torch.long)

            if edge_list.dim() != 2 or edge_list.shape[1] != 2:
                raise ValueError("edge_list must contain (src, dst) pairs.")

    # Extract source and destination indices.
    src = edge_list[:, 0]
    dst = edge_list[:, 1]

    # Infer the number of nodes if it was not provided.
    if num_nodes is None:
        if edge_list.numel() == 0:
            num_nodes = 0
        else:
            num_nodes = int(edge_list.max().item()) + 1
    else:
        num_nodes = int(num_nodes)

    return src, dst, num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # Create self-loop indices using the same dtype as the input tensors.
    loops_src = torch.arange(
        num_nodes,
        dtype=src.dtype,
        device=src.device,
    )

    loops_dst = torch.arange(
        num_nodes,
        dtype=dst.dtype,
        device=dst.device,
    )

    # Append the self-loops to the existing COO edge indices.
    src_out = torch.cat([src, loops_src], dim=0)
    dst_out = torch.cat([dst, loops_dst], dim=0)

    return src_out, dst_out

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # Use edge weights when provided; otherwise, each edge contributes 1.
    if edge_weight is None:
        values = torch.ones(
            src.size(0),
            dtype=torch.float32,
            device=src.device,
        )
    else:
        values = edge_weight.to(dtype=torch.float32, device=src.device)

    # Scatter edge contributions onto their destination nodes.
    degrees = torch.zeros(
        num_nodes,
        dtype=torch.float32,
        device=src.device,
    )

    degrees.scatter_add_(0, dst, values)

    return degrees

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # Default to unit edge weights.
    if edge_weight is None:
        weights = torch.ones(
            src.size(0),
            dtype=torch.float32,
            device=src.device,
        )
    else:
        weights = edge_weight.to(
            dtype=torch.float32,
            device=src.device,
        )

    # Compute weighted IN-degrees.
    # This matches compute_node_degrees(): degree is accumulated
    # at the destination node of each edge.
    degrees = torch.zeros(
        num_nodes,
        dtype=torch.float32,
        device=src.device,
    )

    degrees.scatter_add_(0, dst, weights)

    # Compute d^(-1/2), with zero for zero-degree nodes.
    inv_sqrt_degree = torch.zeros_like(degrees)

    nonzero = degrees > 0
    inv_sqrt_degree[nonzero] = torch.rsqrt(degrees[nonzero])

    # w_ij / sqrt(d_i * d_j)
    normalized_weights = (
        weights
        * inv_sqrt_degree[src]
        * inv_sqrt_degree[dst]
    )

    return normalized_weights

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    """Return edge-aligned source feature rows from node_features.

    Args:
        node_features (Tensor): Node feature matrix of shape (N, F).
        src (LongTensor): Source node indices of shape (E,).

    Returns:
        Tensor: Source features aligned with edges, shape (E, F).
    """
    return node_features[src]

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # Initialize zero output while preserving the input dtype and device.
    node_features = torch.zeros(
        num_nodes,
        edge_features.size(1),
        dtype=edge_features.dtype,
        device=edge_features.device,
    )

    # Scatter-sum each edge feature vector onto its destination node.
    node_features.index_add_(0, dst, edge_features)

    return node_features

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    """Scatter-mean edge features onto destination nodes.

    Args:
        edge_features: Tensor of shape (E, F).
        dst: LongTensor of shape (E,) with destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        Tensor of shape (N, F), with zero vectors for nodes with no incoming edges.
    """
    # Sum edge features at each destination node.
    node_sum = torch.zeros(
        num_nodes,
        edge_features.size(1),
        dtype=edge_features.dtype,
        device=edge_features.device,
    )
    node_sum.index_add_(0, dst, edge_features)

    # Count incoming edges for each node.
    counts = torch.zeros(
        num_nodes,
        dtype=edge_features.dtype,
        device=edge_features.device,
    )
    ones = torch.ones(
        dst.size(0),
        dtype=edge_features.dtype,
        device=edge_features.device,
    )
    counts.index_add_(0, dst, ones)

    # Divide by the number of incoming edges.
    # Clamp avoids division by zero for isolated nodes.
    node_mean = node_sum / counts.clamp_min(1).unsqueeze(1)

    return node_mean

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    """Scatter-max edge features onto destination nodes.

    Args:
        edge_features: Tensor of shape (E, F).
        dst: LongTensor of shape (E,) giving destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        Tensor of shape (N, F), containing elementwise maxima.
        Nodes with no incoming edges contain -inf.
    """
    # Initialize every node-feature entry to -inf.
    node_max = torch.full(
        (num_nodes, edge_features.size(1)),
        float("-inf"),
        dtype=edge_features.dtype,
        device=edge_features.device,
    )

    # Scatter elementwise maxima onto destination nodes.
    for i in range(edge_features.size(0)):
        node = dst[i]
        node_max[node] = torch.maximum(node_max[node], edge_features[i])

    return node_max

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # Gather source and destination node features for every edge.
    src_features = gather_source_node_features(node_features, src)
    dst_features = node_features[dst]

    # Apply the message function with or without edge attributes.
    if edge_attr is None:
        messages = message_fn(src_features, dst_features)
    else:
        messages = message_fn(src_features, dst_features, edge_attr)

    return messages

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    if aggr == 'sum':
        return scatter_sum_to_nodes(messages, dst, num_nodes)
    elif aggr == 'mean':
        return scatter_mean_to_nodes(messages, dst, num_nodes)
    elif aggr == 'max':
        return scatter_max_to_nodes(messages, dst, num_nodes)
    else:
        raise ValueError(
            f"Unsupported aggregation mode: {aggr!r}. "
            "Expected one of {'sum', 'mean', 'max'}."
        )

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    """Update node features using aggregated neighborhood messages.

    Args:
        node_features: Tensor of shape (N, F) containing current node features.
        aggregated: Tensor of shape (N, M) containing aggregated messages.
        update_fn: Callable(node_features, aggregated) -> updated features.

    Returns:
        Tensor containing the updated node features.
    """
    return update_fn(node_features, aggregated)

# Step 12 - message_passing_layer
def message_passing_layer(
    node_features,
    src,
    dst,
    message_fn,
    update_fn,
    aggr='sum',
    edge_attr=None,
):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    # Step 1: Compute one message vector per edge.
    messages = compute_messages(
        node_features,
        src,
        dst,
        message_fn,
        edge_attr=edge_attr,
    )

    # Step 2: Aggregate edge messages onto destination nodes.
    aggregated = aggregate_messages(
        messages,
        dst,
        num_nodes=node_features.size(0),
        aggr=aggr,
    )

    # Step 3: Update each node's feature/state.
    updated_features = update_node_features(
        node_features,
        aggregated,
        update_fn,
    )

    return updated_features

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    h = node_features
    all_layer_outputs = []

    for layer in layers:
        h = layer(h, src, dst, edge_attr=edge_attr)
        all_layer_outputs.append(h)

    return h, all_layer_outputs

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # Add one self-loop (i, i) for every node.
    src_hat, dst_hat = add_self_loops(
        src,
        dst,
        num_nodes,
    )

    # Compute the normalized edge weights on the augmented graph.
    norm_weight = symmetric_normalize_edge_weights(
        src_hat,
        dst_hat,
        num_nodes,
    )

    return src_hat, dst_hat, norm_weight

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    out = node_features @ weight

    if bias is not None:
        out = out + bias

    return out

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    # Add self-loops and compute symmetric normalization weights.
    src_hat, dst_hat, norm_weight = gcn_renormalize_adjacency(
        src, dst, num_nodes
    )

    # Apply the learnable linear transformation.
    transformed = gcn_linear_transform(
        node_features, weight, bias
    )

    # Gather transformed source features for each augmented edge.
    edge_features = transformed[src_hat]

    # Weight each message by the normalized adjacency coefficient.
    edge_features = edge_features * norm_weight.unsqueeze(-1)

    # Aggregate weighted messages at destination nodes.
    out = scatter_sum_to_nodes(
        edge_features, dst_hat, num_nodes
    )

    # Apply the optional activation function.
    if activation is not None:
        out = activation(out)

    return out

# Step 17 - init_gcn_parameters
def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    """Initialize GCN parameters with Glorot-style uniform initialization.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        with_bias: Whether to include a zero-initialized bias.
        seed: Optional PyTorch RNG seed.

    Returns:
        Dictionary containing 'weight' and optionally 'bias'.
    """
    if seed is not None:
        torch.manual_seed(seed)

    a = (6.0 / (in_dim + out_dim)) ** 0.5

    weight = torch.empty(in_dim, out_dim).uniform_(-a, a)

    params = {
        'weight': weight,
    }

    if with_bias:
        params['bias'] = torch.zeros(out_dim)

    return params

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    if activations is None:
        activations = [None] * len(param_list)

    if len(activations) != len(param_list):
        raise ValueError("activations must have one entry per GCN layer.")

    h = node_features
    all_layer_outputs = []

    for params, activation in zip(param_list, activations):
        h = gcn_layer_forward(
            h,
            src,
            dst,
            params["weight"],
            bias=params.get("bias"),
            num_nodes=num_nodes,
            activation=activation,
        )
        all_layer_outputs.append(h)

    return h, all_layer_outputs

# Step 19 - gat_attention_logits
def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # Linear transformation: H' = H @ W.
    transformed = node_features @ weight

    # Gather transformed source and destination node features.
    src_features = transformed[src]
    dst_features = transformed[dst]

    # Compute the two attention contributions for each edge.
    src_scores = (src_features * attn_src).sum(dim=-1)
    dst_scores = (dst_features * attn_dst).sum(dim=-1)

    # Apply the GAT LeakyReLU nonlinearity with slope 0.2.
    logits = torch.nn.functional.leaky_relu(
        src_scores + dst_scores,
        negative_slope=0.2,
    )

    return logits, transformed

# Step 20 - gat_masked_neighbor_softmax
def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # Find the maximum logit for each destination node.
    max_logits = torch.full(
        (num_nodes,),
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )

    max_logits.scatter_reduce_(
        0,
        dst,
        logits,
        reduce="amax",
        include_self=True,
    )

    # Subtract the per-destination maximum for numerical stability.
    stabilized = logits - max_logits[dst]

    # Exponentiate the stabilized logits.
    exp_logits = torch.exp(stabilized)

    # Sum exponentials over incoming edges for each destination.
    denom = torch.zeros(
        num_nodes,
        dtype=logits.dtype,
        device=logits.device,
    )
    denom.scatter_add_(0, dst, exp_logits)

    # Normalize each edge by its destination's sum.
    attention = exp_logits / denom[dst]

    return attention

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    # Compute unnormalized attention logits and transformed node features.
    logits, transformed = gat_attention_logits(
        node_features,
        src,
        dst,
        attn_src,
        attn_dst,
        weight,
    )

    # Normalize attention over incoming neighbors of each destination node.
    attn_coeffs = gat_masked_neighbor_softmax(
        logits,
        dst,
        num_nodes,
    )

    # Weight source-node features by their attention coefficients.
    messages = transformed[src] * attn_coeffs.unsqueeze(-1)

    # Aggregate weighted messages at destination nodes.
    head_out = scatter_sum_to_nodes(
        messages,
        dst,
        num_nodes,
    )

    # Add optional bias.
    if bias is not None:
        head_out = head_out + bias

    # Apply optional activation after bias.
    if activation is not None:
        head_out = activation(head_out)

    return head_out, attn_coeffs

# Step 22 - merge_gat_heads
def merge_gat_heads(head_outputs, mode='concat'):
    """Merge multi-head GAT outputs into one node-feature tensor.

    Args:
        head_outputs: list/tuple of tensors [N, F], or a stacked tensor [H, N, F].
        mode: 'concat' or 'mean'.

    Returns:
        Tensor of shape [N, H*F] for concat or [N, F] for mean.
    """
    if mode not in ('concat', 'mean'):
        raise ValueError(
            f"Unsupported merge mode: {mode!r}. "
            "Expected 'concat' or 'mean'."
        )

    # Convert a list/tuple of heads to [H, N, F].
    if isinstance(head_outputs, (list, tuple)):
        if len(head_outputs) == 0:
            raise ValueError("head_outputs cannot be empty.")

        stacked = torch.stack(head_outputs, dim=0)
    elif isinstance(head_outputs, torch.Tensor):
        if head_outputs.dim() != 3:
            raise ValueError(
                "A stacked head tensor must have shape [H, N, F]."
            )

        stacked = head_outputs
    else:
        raise TypeError(
            "head_outputs must be a list, tuple, or Tensor."
        )

    if mode == 'concat':
        # [H, N, F] -> [N, H, F] -> [N, H*F]
        return stacked.permute(1, 0, 2).reshape(
            stacked.size(1),
            -1,
        )

    # [H, N, F] -> [N, F]
    return stacked.mean(dim=0)

# Step 23 - gat_layer_forward (not yet solved)
# TODO: implement

# Step 24 - init_gat_parameters (not yet solved)
# TODO: implement

# Step 25 - gat_stack_forward (not yet solved)
# TODO: implement

# Step 26 - global_mean_pool (not yet solved)
# TODO: implement

# Step 27 - global_sum_pool (not yet solved)
# TODO: implement

# Step 28 - global_max_pool (not yet solved)
# TODO: implement

# Step 29 - global_mean_max_pool (not yet solved)
# TODO: implement

# Step 30 - node_classification_head (not yet solved)
# TODO: implement

# Step 31 - graph_regression_head (not yet solved)
# TODO: implement

# Step 32 - generate_sbm_graph (not yet solved)
# TODO: implement

# Step 33 - build_node_classification_dataset (not yet solved)
# TODO: implement

# Step 34 - generate_molecule_like_graph (not yet solved)
# TODO: implement

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 38 - mse_loss (not yet solved)
# TODO: implement

# Step 39 - accuracy_metric (not yet solved)
# TODO: implement

# Step 40 - mae_metric (not yet solved)
# TODO: implement

# Step 41 - gnn_train_step (not yet solved)
# TODO: implement

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement


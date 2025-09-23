import torch
import math

def get_mask_pam(input_size, window_size, inner_size):
    """
    Get the attention mask for the PAM-Naive layer in the Pyraformer transformer.
    The mask defines the connections between tokens in different layers and optimizes the attention mechanism by
    limiting the allowed connections.

    Args:
        input_size (int): The size of the input sequence.
        window_size (list of int): The window size of each layer in the PAM-Naive layer.
        inner_size (int): The size of the inner window for the intra-scale mask.

    Returns:
        mask (torch.Tensor, bool): A boolean tensor representing the attention mask for the PAM-Naive layer.
        all_size (list of int): A list of integers representing the size of all the layers in the PAM-Naive layer.
    """

    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length)

    # get intra-scale mask.
    # This mask is responsible for allowing attention within a certain range in the same layer.
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask.
    # This mask is responsible for connecting tokens from one layer to another in the pyramid architecture
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + \
                        (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (
                                     start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


def refer_points(all_sizes, window_size):
    """
    Computes a mapping between input tokens and their corresponding tokens in each layer of the pyramid architecture
    used by the Pyraformer encoder.

    Args:
        all_sizes (list[int]): A list of the number of tokens in each layer of the pyramid architecture.
        window_size (int or list[int]): The size(s) of the sliding window(s) for the bottleneck convolutional CSCM
            layer. If an integer is provided, the same window size is used for all convolutional layers. If a list of
            integers is provided, each convolutional layer uses a different window size.

    Returns:
        Tensor: The mapping between input tokens and their corresponding tokens in each layer of the pyramid
        architecture.
    """
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes))

    # loop through all the tokens in the input sequence and for each token iterate
    # through all the layers in the pyramid architecture
    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + \
                           min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


class RegularMask():
    """
    A utility class for handling attention masks in the Pyraformer architecture.

    Args:
        mask (Tensor): The attention mask tensor.
    """
    def __init__(self, mask):
        self._mask = mask.unsqueeze(1)

    @property
    def mask(self):
        return self._mask


class TriangularCausalMask:
    """
    Generates an upper-triangular binary mask to prevent the model from attending to future time-steps during
    self-attention mechanism.

    Args:
    - B (int): Batch size.
    - L (int): Sequence length.
    - device (str or torch.device): Device on which to create the mask.

    Attributes:
    - mask (torch.Tensor): A tensor of shape (B, 1, L, L) representing the binary mask.
    """

    def __init__(self, B, L, device="cpu"):
        # Create a mask shape of size [B, 1, L, L] where B is the batch size, L is the sequence length
        mask_shape = [B, 1, L, L]

        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Probability mask to be applied to attention scores, where each position i in the mask
        represents whether the ith query is allowed to attend to positions > i in the values.

        Args:

        - B (int): Batch size.
        - H (int): Number of attention heads.
        - L (int): Length of the input queries.
        - index (torch.Tensor): Indices of top-k queries for each batch and head, of shape (B, H, c*ln(L_Q)).
        - scores (torch.Tensor): Attention scores tensor of shape (B, H, L_Q, L_V).
        - device (str or torch.device): Device to be used for tensor operations.

        Returns:
        _mask (torch.Tensor): Probability mask tensor of shape (B, H, L_Q, L_V).
        """
        upper_triangular_mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)

        expanded_mask = upper_triangular_mask[None, None, :].expand(B, H, L, scores.shape[-1])

        indicator = expanded_mask[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)

        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

import torch


def _make_seq_first(*args):
    # N, S, ... -> S, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, 2) if arg is not None else None
    return (*(arg.permute(1, 0, 2) if arg is not None else None for arg in args),)


def _make_batch_first(*args):
    # S, N, ... -> N, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, 2) if arg is not None else None
    return (*(arg.permute(1, 0, 2) if arg is not None else None for arg in args),)


def _get_key_padding_mask(mask):
    key_padding_mask = (mask == 0).cumsum(dim=0) > 0
    return key_padding_mask


def _get_padding_mask(mask):
    mask = _make_seq_first(mask.unsqueeze(2))
    return mask


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _make_group_first(*args):
    # S, G, N, ... -> G, S, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, 2, 3) if arg is not None else None
    return (*(arg.permute(1, 0, 2, 3) if arg is not None else None for arg in args),)


def _pack_group_batch(*args):
    # S, G, N, ... -> S, G * N, ...
    if len(args) == 1:
        arg, = args
        return arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:]) if arg is not None else None
    return (*(arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:]) if arg is not None else None for arg in args),)


def _unpack_group_batch(N, *args):
    # S, G * N, ... -> S, G, N, ...
    if len(args) == 1:
        arg, = args
        return arg.reshape(arg.size(0), -1, N, *arg.shape[2:]) if arg is not None else None
    return (*(arg.reshape(arg.size(0), -1, N, *arg.shape[2:]) if arg is not None else None for arg in args),)

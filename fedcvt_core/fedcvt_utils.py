import torch


def aggregate(guest_reprs, host_reprs, aggregation_mode="cat"):
    if aggregation_mode == "cat":
        fed_reprs = torch.cat([guest_reprs, host_reprs], dim=1)
    elif aggregation_mode == "add":
        fed_reprs = guest_reprs + host_reprs
    else:
        raise Exception("Currently does not support {} aggregation mode.".format(aggregation_mode))
    return fed_reprs


def get_fed_input_dim(host_input_dim, guest_input_dim, aggregation_mode="cat"):
    if aggregation_mode == "cat":
        fed_input_dim = host_input_dim + guest_input_dim
    elif aggregation_mode == "add":
        assert host_input_dim, guest_input_dim
        fed_input_dim = host_input_dim
    else:
        raise Exception("Currently does not support {} aggregation mode.".format(aggregation_mode))
    return fed_input_dim

import tensorflow as tf
import argparse

def get_net_arch(args: argparse.Namespace, is_lstm=False):
    """function to convert input args into valid syntax for the PPO"""
    shared_layers, policy_layers, value_layers = None, None, None

    if args.shared_layers != "":
        shared_layers = parse_network_size(args.shared_layers)
    if args.policy_layer_sizes != "":
        policy_layers = parse_network_size(args.policy_layer_sizes)
    if args.value_layer_sizes != "":
        value_layers = parse_network_size(args.value_layer_sizes)

    if shared_layers is None:
        shared_layers = []
    vf_pi = {}
    if value_layers is not None:
        vf_pi["vf"] = value_layers
    if policy_layers is not None:
        vf_pi["pi"] = policy_layers

    if is_lstm:
        net_arch = shared_layers + ['lstm'] + [vf_pi]
    else:
        net_arch = shared_layers + [vf_pi]
    #print(net_arch)
    return net_arch


def parse_network_size(string: str):
    """function to convert a string into a int list

    Example:

    Input: parse_string("64-64")
    Output: [64, 64]

    """
    string_arr = string.split("-")
    int_list = []
    for string in string_arr:
        try:
            int_list.append(int(string))
        except:
            raise Exception("Invalid argument format on: " + string)
    return int_list


def get_act_fn(act_fn_string: str):
    """function to convert str into pytorch activation function class"""
    if act_fn_string == "relu":
        return tf.keras.activations.relu
    elif act_fn_string == "sigmoid":
        return tf.keras.activations.sigmoid
    elif act_fn_string == "tanh":
        return tf.keras.activations.tanh


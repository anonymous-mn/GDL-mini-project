import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm

# mjn
import random

def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


# mjn - return function that flips the y values of a dataset
def flip_y(num_classes):
    def flip_y_func(data):
        np_array_y = data.y.numpy()
        np_array_y[0] = num_classes - np_array_y[0]
        return data
    return flip_y_func

# mjn - if Y is the label of data, set y[i]=0 for all i in mask (i.e., maks is a subset of {0, ..., dim(y)-1}). 
def mask_features(mask):
    def mask_features_func(data):
        for i in mask:
            data.x[:, i] = 0
        return data
    return mask_features_func

# mjn - return function that randomly removes edges according to "Deep Graph Contrastive Representation Learning"
def remove_edges(p=0.4):
    def remove_edges_func(data):
        # Convert `edge_index` to a list of tuples for easier processing
        edge_list = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    
        # Filter out the edges that need to be removed
        edges_to_remove = []
        for edge in edge_list:
            if random.random() < p:
                edges_to_remove.append(edge)
        # For each edge to remove, also add its reverse to the list
        reverse = []
        for edge in edges_to_remove:
            a, b = edge
            reverse.append([b, a])
        edges_to_remove_with_reverse = edges_to_remove + reverse
    
        # Filter out the edges that need to be removed
        edge_list = [edge for edge in edge_list if edge not in edges_to_remove_with_reverse]
            
        # Convert back to a tensor format
        filtered_edge_index = torch.tensor(edge_list).t().contiguous()
        
        # Update the `Data` object
        data.edge_index = filtered_edge_index
        return data
        #for idx in range(data.edge_index.shape[1]):
        #    if random.random() < p:
        #        data.remove_edge_index(idx)
    return remove_edges_func


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data

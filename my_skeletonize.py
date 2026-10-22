import os
import numpy as np

from skimage.morphology import binary_closing, ball, skeletonize
from scipy.ndimage import (
    binary_fill_holes, 
    label, map_coordinates, 
    binary_dilation, 
    binary_erosion,
    generate_binary_structure,
    affine_transform
)
import networkx as nx
from itertools import product



def build_graph(skeleton):
    G = nx.Graph()

    skeleton_voxels = np.argwhere(skeleton > 0)
    voxel_set = set(map(tuple, skeleton_voxels)) # transform to set of tuples for fast lookup

    # Define 26-connectivity offsets
    offsets = list(product([-1, 0, 1], repeat=3))
    offsets.remove((0, 0, 0)) # remove the center voxel

    # Build edges
    for voxel in skeleton_voxels:
        voxel_tuple = tuple(voxel)
        for offset in offsets:
            neighbor = tuple(voxel + offset)
            if neighbor in voxel_set:
                G.add_edge(voxel_tuple, neighbor)
    return G

def collapse_junction_clusters(G_ori, min_cluster_size=4, min_degree=3):
    G = G_ori.copy()
    junction_nodes = [n for n in G.nodes() if G.degree(n) >= min_degree]
    J = G.subgraph(junction_nodes)
    clusters = list(nx.connected_components(J))
    print("Junction cluster sizes:", sorted([len(c) for c in clusters], reverse=True)[:20], flush=True)

    collapse_clusters = [set(cluster) for cluster in clusters if len(cluster) >= min_cluster_size]
    print(f"Clusters to collapse: {len(collapse_clusters)}", flush=True)

    if len(collapse_clusters) == 0:
        return G

    node_to_rep = {}
    representatives = []

    for collapse_idx, cluster in enumerate(collapse_clusters):
        pts = np.asarray(list(cluster), dtype=float)
        centroid = np.round(pts.mean(axis=0)).astype(int)

        # Choose an EXISTING skeleton node as representative.
        representative = min(cluster,
            key=lambda n: np.linalg.norm(
                np.asarray(n, dtype=float) -
                np.asarray(centroid, dtype=float)
            ))
        representatives.append(representative)

        # Every node in this cluster maps to representative
        for node in cluster:
            node_to_rep[node] = representative

        # print(f"Cluster {collapse_idx}: {len(cluster)} nodes -> {representative}, "
        #     f"centroid={tuple(centroid)}", flush=True)

    # 4. Rebuild graph using the mapping
    G_new = nx.Graph()

    # ---- Add nodes ----
    for node, data in G.nodes(data=True):
        representative = node_to_rep.get(node, node)
        # If this representative has not been added yet
        if representative not in G_new:
            # Prefer attributes of the representative itself
            rep_data = G.nodes[representative].copy()
            G_new.add_node(representative, **rep_data)

    # 5. Re-map EVERY original edge
    for u, v, data in G.edges(data=True):
        new_u = node_to_rep.get(u, u)
        new_v = node_to_rep.get(v, v)

        # Edge entirely inside the same junction cluster
        # becomes a self-loop -> discard it.
        if new_u == new_v:
            continue

        # Add the collapsed edge
        G_new.add_edge(new_u, new_v,**data)

    # 6. Debug / validation
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    print(f"Collapsed graph: {G_new.number_of_nodes()} nodes, {G_new.number_of_edges()} edges", flush=True)

    # Check connectivity
    n_components_before = nx.number_connected_components(G)
    n_components_after = nx.number_connected_components(G_new)

    print(f"Connected components: {n_components_before} -> {n_components_after}", flush=True)

    if n_components_after > n_components_before:
        print("WARNING: collapse increased the number of connected components!", flush=True)

    # Print final degree of representatives
    # for rep in representatives:
    #     if rep in G_new:
    #         print(f"Representative {rep}: degree={G_new.degree(rep)}", flush=True)

    print(f"Collapsed {len(collapse_clusters)} junction clusters", flush=True)

    return G_new

def remove_small_cycles(G, max_cycle_size=3):
    cycles = nx.cycle_basis(G)
    removed = 0

    for cycle in cycles:
        if len(cycle) > max_cycle_size:
            continue
        edges = [(cycle[i], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))]
        lengths = [np.linalg.norm(np.array(u)-np.array(v)) for u,v in edges]
        edge_to_remove = edges[np.argmin(lengths)]
        if G.has_edge(*edge_to_remove):
            G.remove_edge(*edge_to_remove)
            removed += 1

    print(f"Removed {removed} small cycles")
    return G

def graph_to_voxel(G, shape):
    voxel_map = np.zeros(shape, dtype=np.uint8)
    for node in G.nodes():
        voxel_map[node] = 1
    return voxel_map

def skeletonize_clean(binary_mask):
    skeleton = skeletonize(binary_mask)
    
    # Remove bunky clusters
    G = build_graph(skeleton)
    G = collapse_junction_clusters(G, min_cluster_size=4)
    G = remove_small_cycles(G)

    # Convert back to voxel representation
    skeleton_cleaned = graph_to_voxel(G, skeleton.shape)
    return skeleton_cleaned

def skeletonize_to_voxels(binary_mask):
    skeleton = skeletonize_clean(binary_mask)
    voxel_map = np.zeros_like(binary_mask, dtype=np.uint8)
    skeleton_voxels = np.argwhere(skeleton > 0)
    for voxel in skeleton_voxels:
        voxel_map[tuple(voxel)] = 1
    return voxel_map
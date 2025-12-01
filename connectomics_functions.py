import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import networkx as nx
from IPython.display import IFrame, display
import tempfile, webbrowser, os

def plot_adjacency_matrix(G, sort_nodes=True, cmap="jet", step=1, figsize=(2,2)):
    """
    Plot adjacency matrix with zeros shown as white.
    """
    # Optionally sort nodes by degree
    if sort_nodes:
        nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    else:
        nodes = list(G.nodes())

    # Create adjacency matrix
    A = nx.to_numpy_array(G, nodelist=nodes, weight="Weight")  
    # Mask zeros so they appear white
    A_masked = np.ma.masked_where(A == 0, A)

    # Load the requested colormap correctly
    cmap_mod = plt.colormaps[cmap].copy()   # <-- FIXED
    cmap_mod.set_bad(color="white")          # masked = white

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    im = ax.imshow(A_masked, interpolation="nearest", cmap=cmap_mod)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight")

    # Ticks
    ax.set_xticks(np.arange(0, len(nodes), step))
    ax.set_yticks(np.arange(0, len(nodes), step))
    ax.set_xticklabels(nodes[::step], rotation=90, fontsize=4)
    ax.set_yticklabels(nodes[::step], fontsize=4)

    ax.set_xlabel("Neurons", fontsize=6)
    ax.set_ylabel("Neurons", fontsize=6)
    ax.set_title("Adjacency Matrix - Weights", fontsize=7)

    plt.tight_layout()
    plt.show()


def average_connection_distance(G, pos):
    dist = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            dists = [np.linalg.norm(np.array(pos[node]) - np.array(pos[n]))
                     for n in neighbors if n in pos]
            dist[node] = np.mean(dists) if dists else np.nan
        else:
            dist[node] = np.nan
    return dist



def nodal_efficiency(G):
    eff = {}
    length_dict = dict(nx.all_pairs_shortest_path_length(G))
    for node in G.nodes():
        L = [length_dict[node][other]
             for other in G.nodes() if other != node and other in length_dict[node]]
        eff[node] = np.mean([1/l for l in L if l > 0]) if L else 0
    return eff


def average_connection_distance(G, pos):
    dist = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            dists = [np.linalg.norm(np.array(pos[node]) - np.array(pos[n]))
                     for n in neighbors if n in pos]
            dist[node] = np.mean(dists) if dists else np.nan
        else:
            dist[node] = np.nan
    return dist


def nodal_efficiency(G):
    eff = {}
    length_dict = dict(nx.all_pairs_shortest_path_length(G))
    for node in G.nodes():
        L = [length_dict[node][other]
             for other in G.nodes() if other != node and other in length_dict[node]]
        eff[node] = np.mean([1/l for l in L if l > 0]) if L else 0
    return eff

def participation_coefficient(G, modules):
    pc = {}
    for node in G.nodes():
        ki = G.degree(node)
        if ki == 0:
            pc[node] = 0
            continue
        sum_frac_sq = 0
        for m in set(modules.values()):
            kis = sum(1 for nbr in G.neighbors(node) if modules[nbr] == m)
            sum_frac_sq += (kis / ki) ** 2
        pc[node] = 1 - sum_frac_sq
    return pc


def metric_set(G):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if nx.is_connected(G):
        Gc = G
    else:
        Gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    metrics = {}
    degs = np.array([d for _, d in G.degree()])
    metrics['mean_degree'] = degs.mean()
    metrics['deg_var'] = degs.var()
    metrics['max_degree'] = degs.max()
    metrics['transitivity'] = nx.transitivity(G)
    metrics['triangle_count'] = sum(nx.triangles(G).values()) / 3
    metrics['assortativity'] = nx.degree_pearson_correlation_coefficient(G)
    try:
        metrics['avg_path_length'] = nx.average_shortest_path_length(Gc)
    except Exception:
        metrics['avg_path_length'] = np.nan

    # spectral radius (largest eigenvalue)
    from scipy.sparse.linalg import eigs
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)  # updated call
    try:
        vals, _ = eigs(A, k=1, which='LR')
        metrics['spectral_radius'] = np.real(vals[0])
    except Exception:
        metrics['spectral_radius'] = np.nan

    return metrics




def randomize_graph(G, method="degree_preserving", nswap=10,swap_factor=100, max_tries_factor=10, max_tries=10000,seed=None):
    """
    Generate a randomized version of graph G.
    """
    E = G.number_of_edges()
    
    if method == "degree_preserving":
        G0 = nx.Graph(G)          # converts DiGraph/MultiGraph -> Graph
        G0.remove_edges_from(nx.selfloop_edges(G0))
        M = G0.number_of_edges()
        nswap = swap_factor * M
        max_tries = max( nswap * max_tries_factor, nswap + 1000 )  # give plenty of attempts

        G_rand = G0.copy()
        nx.double_edge_swap(G_rand, nswap=nswap, max_tries=max_tries, seed=seed)

        # guard: remove any self-loops just in case
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    
    
    elif method == "erdos_renyi":
        n = G.number_of_nodes()
        m = G.number_of_edges()
        p = m / (n * (n - 1) / 2)  # probability for ER graph with same density
        G_rand = nx.erdos_renyi_graph(n, p)
    else:
        raise ValueError("method must be 'degree_preserving' or 'erdos_renyi'")
    return G_rand


def rich_club(G, method="degree_preserving", k_max=None,
              nswap=10, max_tries=100000, n_rand=100):
    """
    Compute rich-club coefficients and normalized values.
    Handles directed graphs and self-loops by simplifying.
    """


    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G_und = G.to_undirected()
    
    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.richclub.rich_club_coefficient.html
    rc_real = nx.rich_club_coefficient(G_und, normalized=False,Q=None)
    
    if k_max is None:
        k_max = max(dict(G_und.degree()).values())


    # collect random rich-club coefficients
    rc_rand_list = []
    for _ in range(n_rand):
        G_rand = randomize_graph(G_und, method=method, nswap=nswap, max_tries=max_tries)

        if G_rand.is_directed() or G_rand.is_multigraph():
            G_rand = nx.Graph(G_rand)

        # remove self-loops in randomized graphs too
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

        rc_rand = nx.rich_club_coefficient(G_rand, normalized=False, Q=None)
        rc_rand_list.append(rc_rand)

    # compute mean and std for each k
    rc_rand_mean = {}
    rc_rand_std = {}
    for k in range(k_max + 1):
        values = [rc[k] for rc in rc_rand_list if k in rc]
        if values:
            rc_rand_mean[k] = np.mean(values)
            rc_rand_std[k] = np.std(values)
        else:
            rc_rand_mean[k] = np.nan
            rc_rand_std[k] = np.nan

    # normalized coefficients
    rc_norm = {k: (rc_real.get(k, np.nan) / rc_rand_mean[k]
                   if rc_rand_mean[k] and rc_rand_mean[k] > 0 else np.nan)
               for k in range(k_max + 1)}

    return rc_real, rc_rand_mean, rc_rand_std, rc_norm


def plot_rich_club(rc_real,rc_rand_avg,rc_rand_std,rc_norm,k_vals,k_rich):
    # ======== PLOT REAL vs RANDOMIZED ========
    fig, ax = plt.subplots(dpi=600)
    ax.plot(k_vals, list(rc_real.values()), '.-', color='red', label='Real φ(k)')

    rand_vals = np.array([rc_rand_avg[k] for k in k_vals])
    rand_std = np.array([rc_rand_std[k] for k in k_vals])
    ax.plot(k_vals, rand_vals, '.-', color='grey', label='Randomized φ(k)')
    ax.fill_between(k_vals, rand_vals - rand_std, rand_vals + rand_std, color='grey', alpha=0.3, label='Rand ± STD')

    if k_rich is not None:
        ax.axvline(k_rich, color='blue', linestyle='--', label='Rich-club threshold')

    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('Rich-Club Coefficient')
    ax.set_title('Rich-Club Coefficient (C. elegans)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # ======== PLOT NORMALIZED φ(k) ========
    fig, ax = plt.subplots(dpi=600)
    norm_vals = np.array([rc_norm[k] for k in k_vals])
    rand_std_norm = np.array([rc_rand_std[k] / rc_rand_avg[k] if rc_rand_avg[k] > 0 else 0 for k in k_vals])

    ax.plot(k_vals, norm_vals, '.-', color='green', label='Normalized φ(k)')
    ax.fill_between(k_vals, norm_vals - rand_std_norm, norm_vals + rand_std_norm, color='green', alpha=0.3, label='Rand ± STD')

    if k_rich is not None:
        ax.axvline(k_rich, color='blue', linestyle='--', label='Rich-club threshold')

    ax.axhline(1.0, linestyle='--', color='red')
    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('Normalized Rich-Club Coefficient')
    ax.set_title('Normalized Rich-Club Coefficient (C. elegans)')
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_weight_distribution(G, bins=50, fit_powerlaw=True):
    """
    Extracts synaptic weights from a NetworkX graph and plots:
    - histogram of weights
    - optional power-law fit on log-log scale
    """
    # Extract all weights (skip zeros and missing)
    weights = np.array([
        d.get("Weight", d.get("weight"))
        for _, _, d in G.edges(data=True)
        if d.get("Weight", d.get("weight")) is not None
    ])
    weights = weights[weights > 0]

    if len(weights) == 0:
        print("No positive weights found.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

    # --- Histogram (linear scale)
    ax[0].hist(weights, bins=bins, edgecolor="black")
    ax[0].set_title("Synaptic Weight Histogram")
    ax[0].set_xlabel("Weight")
    ax[0].set_ylabel("Count")

    # --- Log-log distribution
    ax[1].scatter(weights, np.ones_like(weights), s=1, alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_title("Log-Log Weight Distribution")
    ax[1].set_xlabel("Weight")

    # --- Fit power law
    if fit_powerlaw:
        try:
            import powerlaw
            fit = powerlaw.Fit(weights, verbose=False)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            ax[1].plot(
                np.linspace(xmin, weights.max(), 200),
                (np.linspace(xmin, weights.max(), 200) ** -alpha) * len(weights),
                label=f"Power-law fit (α={alpha:.2f})"
            )
            ax[1].legend()
        except ImportError:
            # fallback: linear fit in log-log space
            log_w = np.log10(weights)
            y = np.arange(len(weights)) + 1
            coeff = np.polyfit(log_w, np.log10(y), 1)
            ax[1].plot(weights, 10 ** (coeff[1] + coeff[0]*log_w),
                       label=f"Log-log fit slope={coeff[0]:.2f}")
            ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_network(G, node_size_factor=30, edge_width_factor=2, cmap="viridis", figsize=(5,5)):
    """
    Draws the network using matplotlib:
    - node size proportional to degree
    - edge width proportional to weight
    - colored edges
    """
    # Extract weights
    weights = np.array([
        d.get("Weight", d.get("weight", 1))
        for _, _, d in G.edges(data=True)
    ])

    # Normalize edge colors to [0,1]
    if len(weights) == 0:
        weights_norm = np.ones(len(G.edges()))
    else:
        w_min, w_max = weights.min(), weights.max()
        weights_norm = (weights - w_min) / (w_max - w_min + 1e-9)

    pos = nx.spring_layout(G, seed=42)

    node_sizes = [G.degree(n) * node_size_factor for n in G.nodes()]

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="black",
        alpha=0.8
    )
    nx.draw_networkx_edges(
        G, pos,
        width=weights * edge_width_factor,
        edge_color=weights_norm,
        edge_cmap=plt.get_cmap(cmap),
        alpha=0.7
    )

    plt.axis("off")
    plt.title("Network Graph (weighted)")
    plt.show()


def visualize_nx_graph_inline(G,
                              physics=True,
                              node_size=30,
                              edge_width_factor=1,
                              label_font_size=20,
                              weight_label="Weight"):
    
    net = Network(height="800px", width="100%", directed=G.is_directed(), notebook=True)

    # Physics
    if physics:
        net.barnes_hut()
    else:
        net.force_atlas_2based(gravity=-100)
        net.toggle_physics(False)

    # Nodes
    for node in G.nodes():
        net.add_node(
            node,
            size=node_size,
            label=str(node),
            title=str(node),
            font={'size': label_font_size, 'align': 'center'},
        )

    # Edges
    for u, v, data in G.edges(data=True):
        w = data.get(weight_label, 1)
        net.add_edge(
            u, v,
            width=w * edge_width_factor,
            arrows="to",
            smooth=True
        )

    # Direct inline display — NO FILE SAVED
    return net.show("graph.html", notebook=True)

        
print("Imports are sucessufl #######################################")






import numpy as np

from scipy.spatial.distance import pdist, squareform

from sklearn.metrics.pairwise import haversine_distances

from tslearn.metrics import dtw

from Utils import VisualUtils

def graphs(
        data: np.ndarray,
        locations: np.ndarray,
        q: int = 75
    ):
    """
    Calculates the graphs related to the dataset

    Parameters:
        data (np.ndarray): The annual GDP per capita data
        locations (np.ndarray): The locations of the countries
        q (int): The quantile of time series distances up to which the edge is kept
        
    Returns:
        np.ndarray: The thresholded weighted adjacency matrix of the DTW distances of time series
        np.ndarray: The weighted adjacency matrix of the geographical distances of the countries
    """    
    pairwise = pdist(data, metric=dtw)
    graph = squareform(pairwise)
    u = np.unique(graph)
    thres = np.percentile(u, q)
    W = np.where(graph < thres, graph, 0)
    W = W/np.linalg.norm(W)
    distances = haversine_distances(locations)

    return W, distances

def graph_edges(
        graph: np.ndarray,
        nodes: list[str]
    ) -> list:
    """
    Returns the list (u, v, w) of nonzero edges in the graph

    Parameters:
        graph (np.ndarray): The weighted adjacency matrix of the graph
        nodes (list[str]): The list of nodes of the graph

    Returns:
    list: The list of the edges as tuples (u, v, w), where u and v are the nodes and w is the weight of the edge
    """
    edges = []
    m = len(nodes)
    for i in range(m):
        for j in range(i+1, m):
            if graph[i][j] > 0:
                edges.append((nodes[i], nodes[j], graph[i][j]))
    return edges

def isolate_group(
        countries: list[str],
        group_members: list[str],
        graph: np.ndarray
    ) -> np.ndarray:
    """
    Isolates a group of countries from the graph containing all countries

    Parameters:
        countries (list[str]): The list of ISO3 codes for each country in the dataset
        group_members (list[str]): The ISO3 codes of the countries to be isolated
        graph (np.ndarray): The weighted adjacency matrix of the global graph
    
    Returns:
        np.ndarray: The m by m matrix containing only the specified countries, where m = len(group_members)
    """
    indices = np.array([countries.index(member) for member in group_members])
    return graph[np.ix_(indices, indices)]


def edge_details(
        edges: list, 
        y: np.ndarray,
        countries: list[str], 
        distances: np.ndarray, 
        min_thickness: float=0.5, 
        max_thickness: float=5
    ):
    """
    Calculates the color and thickness of graph edges. An edge is green if its nodes are in the same cluster; red otherwise. The thickness of an edge is determined by the geographical distance of the node countries

    Parameters:
        edges (list): The list of the edges as tuples (u, v, w), where u and v are the nodes and w is the weight of the edge
        y (np.ndarray): The clustering labels. These determine the edge colors
        countries (list[str]): The list of ISO3 codes for each country in the dataset
        distances (np.ndarray): The graph of the geographical distances
        min_thickness (float): The minimum thickness of edges
        max_thickness (float): The maximum thickness of edges

    Returns:
        list[str]: The list of colors of the edges
        list[float]: The list of thickness of the edges
    """
    colors = []
    thickness = []

    for edge in edges:
        ind1 = countries.index(edge[0])
        ind2 = countries.index(edge[1])
        colors.append('lime' if y[ind1] == y[ind2] else 'red')
        thickness.append(distances[ind1][ind2])
    thin = min(thickness)
    thick = max(thickness)
    thickness = min_thickness+(thickness-thin)*(max_thickness-min_thickness)/(thick-thin)

    return colors, thickness

def postprocess_clustering(
        data: np.ndarray,
        locations: np.ndarray,
        countries: list[str],
        groups: list,
        y: np.ndarray,
        q: float = 75,
        min_thickness: float = 0.5,
        max_thickness: float = 5
    ):
    """
    Post-processes clustering results to produce group specific graphs.

    Parameters:
        data (np.ndarray): The annual GDP per capita data
        locations (np.ndarray): The locations of the countries
        countries (list[str]): The list of ISO3 codes for each country in the dataset
        groups (list): The list containing the group name and group members for each group
        y (np.ndarray): The clustering labels. These determine the edge colors
        q (int): The quantile of time series distances up to which the DTW edge is kept
        min_thickness (float): The minimum thickness of edges
        max_thickness (float): The maximum thickness of edges
    """
    W, distances = graphs(data, locations, q)

    for group in groups:
        title = group[0]
        members = group[1]
        graph = isolate_group(countries, members, W)
        edges = graph_edges(graph, members)
        colors, thickness = edge_details(edges, y, countries, distances, min_thickness, max_thickness)

        VisualUtils.plot_group_graph(members, edges, colors, thickness, title)



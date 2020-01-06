import random
import numpy as np
from multiprocessing import Pool

def walker(w, G, node_list):
    walks = []
    for node in node_list:
        walk = []
        last = node
        while len(walk)<w:
            walk.append(int(last))
            last = random.sample(G[last]["edge"],1)[0]
        walks.append(walk)
    return walks

def subgraph(k, G, node_list):
    subgraphs = []
    for node in node_list:
        a_subgraph = []
        nei_set = random.sample(G[node]["edge"], min(k, len(G[node]["edge"])))
        while len(nei_set)<k:
            nei_set.append(node)
        for nei in nei_set:
            ten_sub = random.sample(G[nei]["edge"], min(k, len(G[nei]["edge"])))
            while len(ten_sub)<k:
                ten_sub.append(node)
            a_subgraph.extend(ten_sub)
        subgraphs.append(a_subgraph)
    return subgraphs

def random_walk(hp, G, node_list):
    walkers = hp.thread
    per_threads_node = len(node_list) // walkers
    # 创建新线程
    results = []
    pool = Pool(processes=walkers)
    for i in range(walkers):
        if i == walkers - 1:
            results.append(
                pool.apply_async(walker, (hp.w, G, node_list[per_threads_node * i:])))
        else:
            results.append(pool.apply_async(walker, (
            hp.w, G, node_list[per_threads_node * i:per_threads_node * (i + 1)])))
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    walks = []
    for it in results:
        for jk in it:
            walks.append(jk)
    return walks

def aggregate(hp, G, node_list):
    walkers = hp.thread
    per_threads_node = len(node_list) // walkers
    # 创建新线程
    results = []
    pool = Pool(processes=walkers)
    for i in range(walkers):
        if i == walkers - 1:
            results.append(
                pool.apply_async(subgraph, (hp.k, G, node_list[per_threads_node * i:])))
        else:
            results.append(pool.apply_async(subgraph, (
                hp.k, G, node_list[per_threads_node * i:per_threads_node * (i + 1)])))
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    subgraph_set = []
    for it in results:
        for jk in it:
            subgraph_set.append(jk)
    return subgraph_set
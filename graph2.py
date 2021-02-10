INF = float("inf")
class Graph2:
    # 無向または有向の、重み付きグラフ
    # 多重辺も記録される
    def __init__(self, n, directed = False, edges = []):
        self.n = n
        self.directed = directed
        self.edges = [[] for _ in range(self.n)]
        for x, y, cost in edges:
            self.add_edge(x, y, cost)

    def add_edge(self, x, y, cost):
        self.edges[x].append((y, cost))
        if self.directed == False:
            self.edges[y].append((x, cost))

    def to_queue(self):
        G = [[INF] * self.n for _ in range(self.n)]
        for i, g in enumerate(self.edges):
            for j, cost in g:
                G[i][j] = cost
        for i in range(self.n):
            G[i][i] = 0
        return G

    def dijkstra(self, start = 0) -> list:
        """
        Dijkstra法 (枝刈り済)
        O(ElogV) ?
        負コストの辺を含まないグラフにのみ使える
        単一始点最短経路一覧
        * heqpq の比較のための key は第一引数である点に注意( = heappush(heapq, (key,value)) )
        """
        res = [INF] * self.n
        res[start] = 0
        next_set = [(0, start)]
        while next_set:
            dist, p = heappop(next_set)
            if res[p] < dist:
                continue
            """ここで頂点pまでの最短距離が確定。よって、ここを通るのはN回のみ"""
            for q, cost in self.edges[p]:
                temp_d = dist + cost
                if temp_d < res[q]:
                    res[q] = temp_d
                    heappush(next_set, (temp_d, q))
        return res

    def shortest_path_faster_algorithm(self, start = 0) -> list:
        """
        更新があるところだけ更新する Bellman-Ford法
        O(VE) だが実用上高速
        負コストの辺があるときでも使える
        単一始点最短経路一覧
        """
        q = [start]
        distance = [INF] * self.n;  distance[start] = 0
        in_q = [0] * self.n;        in_q[start] = True
        times = [0] * self.n;       times[start] = 1
        while q:
            v = q.pop()
            in_q[v] = False
            dist_v = distance[v]
            for u, cost in self.edges[v]:
                new_dist_u = dist_v + cost
                if distance[u] > new_dist_u:
                    times[u] += 1
                    if times[u] >= N:  # 負閉路検出
                        distance[u] = -iINF
                    else:
                        distance[u] = new_dist_u
                    if not in_q[u]:
                        in_q[u] = True
                        q.append(u)
        return distance
    
    def johnson(self):
        return
    
    def warshallfloyd(self, scipy_use = True):
        """
        Warshall-Floyd法
        O(V^3)
        負コストの辺があるときでも使える
        全点間最短経路
        """
        G = self.to_queue()
        import sys
        if scipy_use == False or "PyPy" in sys.version:
            for k in range(self.n):
                for i in range(self.n):
                    for j in range(self.n):
                        if G[i][j] == INF:
                            G[i][j] = G[i][k] + G[k][j]
            return G
        else:
            import numpy as np
            from scipy.sparse.csgraph import floyd_warshall
            # from scipy.sparse import csr_matrix
            # G = csr_matrix(G)
            return floyd_warshall(G)
        

    def Astar(self, start, goal,  h_cost: "func", max_cost = None) -> int:
        """
        Dijkstra法 の延長による A*探索法
        負コストの辺を含まないグラフにのみ使える
        単一始点最短経路
        """
        from collections import defaultdict
        if max_cost is None:
            max_cost = self.n
        g_costs = defaultdict(lambda: INF)
        f_costs = defaultdict(lambda: INF)
        f_costs[start] = h_cost(start, goal)
        next_set = [(f_costs[start], start)]
        while next_set:
            f_cost, p = heappop(next_set)
            if p == goal:
                return g_costs[p]
            if f_costs[p] < f_cost:
                continue
            """ここで頂点pまでの最短距離が確定。よって、ここを通るのはN回のみ"""
            for q, cost in self.edges[p]:
                next_g_cost = g_costs[p] + cost
                next_f_cost = g_costs[p] + cost + h_cost(q, goal)
                if next_f_cost > max_cost:
                    continue
                if next_f_cost < f_costs[q]:
                    g_costs[q] = next_g_cost
                    f_costs[q] = next_f_cost
                    heappush(next_set, (next_f_cost, q))
        return -1
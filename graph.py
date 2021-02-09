class Graph:
    def __init__(self, n, directed = False, destroy = False, edges = []):
        self.n = n
        self.directed = directed
        self.destroy = destroy
        self.parent = [-1] * n
        self.edges = [set() for _ in self.parent]
        for x, y in edges:
            self.add_edge(x, y)

    def add_edge(self, x, y):
        self.edges[x].add(y)
        if self.directed == False:
            self.edges[y].add(x)

    def add_adjacent_list(self, i, adjacent_list):
        self.edges[i] = set(adjacent_list)

    def bfs(self, start = 0, goal = -1, time = 0, save = False):
        """
        :param start: スタート地点
        :param goal: ゴール地点
        :param save: True = 前回の探索結果を保持する
        :return: (ループがあっても)最短距離。存在しなければ -1
        """
        if not save:
            self.parent = [-1] * self.n
        p, t = start, time
        self.parent[p] = -2
        next_set = deque([(p, t)])

        while next_set:
            p, t = next_set.popleft()
            for q in self.edges[p]:
                if q == self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                if self.parent[q] != -1:
                    """ サイクル時の処理 """
                    continue
                if q == goal:
                    self.parent[q]=p
                    return t + 1
                self.parent[q] = p
                next_set.append((q, t + 1))
        return -1

    def connection_counter(self):
        """
        :return: 連結成分の個数。有効グラフではあまり意味がない。
        """
        cnt = 0
        self.parent = [-1] * self.n
        for start in range(self.n):
            if self.parent[start] == -1:
                cnt += 1
                self.bfs(start, save = True)
        return cnt

    def connection_detail(self):
        """
        :return: 連結成分の(頂点数, 辺の数)。有効グラフではあまり意味がない。
        備考: 木であるための必要十分条件は M=N-1
        """
        ver_edge=[]
        self.parent = [-1] * self.n
        for start in range(self.n):
            if self.parent[start] == -1:
                ver_cnt, edge_cnt = self._detail(start)
                ver_edge.append((ver_cnt, edge_cnt))
        return ver_edge

    def _detail(self, start = 0):
        """
        :param start: スタート地点
        :param save: True = 前回の探索結果を保持する
        """
        p, t = start, 0
        self.parent[p] = -2
        next_set = deque([(p, t)])
        sub_edges,sub_vers = set(),set()

        while next_set:
            p, t = next_set.popleft()
            sub_vers.add(p)
            for q in self.edges[p]:
                if q == self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                sub_edges.add((min(p,q),max(p,q)))
                if self.parent[q] != -1:
                    """ サイクル時の処理 """
                    continue
                self.parent[q] = p
                next_set.append((q, t + 1))
        return len(sub_vers), len(sub_edges)

    def distance_list(self, start = 0, save = False):
        """
        :param start: スタート地点
        :return: スタート地点から各点への距離のリスト
        * 距離無限大が -1 になっている点に注意！！！
        """
        dist = [-1] * self.n
        if not save:
            self.parent = [-1] * self.n
        p, t = start, 0
        self.parent[p] = -2
        dist[p] = 0
        next_set = deque([(p, t)])

        while next_set:
            p, t = next_set.popleft()
            for q in self.edges[p]:
                if q == self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                if self.parent[q] != -1:
                    """ サイクル時の処理 """
                    continue
                dist[q] = t + 1
                self.parent[q] = p
                next_set.append((q, t + 1))
        return dist

    def parent_list(self, start = 0):
        """
        :return: スタート地点から最短経路で進んだ時の、各頂点の一個前に訪問する頂点番号
                訪問しない場合は -1 を返す
        """
        self.distance_list(start)
        return list(p for p in self.parent[1:])

    def most_distant_point(self, start = 0, save = False):
        """
        計算量 O(N)
        :return: (start から最も遠い頂点, 距離)
        """
        if not save:
            self.parent = [-1] * self.n
        res = (start, 0)
        temp = 0
        for i, dist in enumerate(self.distance_list(start, save=save)):
            if dist > temp:
                temp = dist
                res = (i, dist)
        return res

    def diameter(self, save = False):
        """
        計算量 O(N)
        :return: 木の直径(最も離れた二頂点間の距離)を返す
        """
        if not save:
            self.parent = [-1] * self.n
        p = self.most_distant_point(save = save)
        res = self.most_distant_point(start = p[0], save = save)
        return res[1]

    def dfs(self, start = 0, goal = -1, time = 0, save = False):
        """
        :param start: スタート地点
        :param goal: ゴール地点
        :param save: True = 前回の探索結果を保持する
        :return: ゴール地点までの距離。存在しなければ -1。ループがある時は最短距離とは限らないため注意。
        """

        if not save:
            self.parent = [-1] * self.n
        if self.destroy:
            edges2 = self.edges
        else:
            edges2 = [self.edges[p].copy() for p in range(self.n)]

        parent,directed=self.parent,self.directed

        p, t = start, time
        parent[p] = -2
        while True:
            if edges2[p]:
                q = edges2[p].pop()
                if q == parent[p] and not directed:
                    """ 逆流した時の処理 """
                    continue
                if parent[q] != -1:
                    """ サイクルで同一点を訪れた時の処理 """
                    continue
                if q == goal:
                    """ ゴール時の処理"""
                    parent[q]=p
                    return t + 1
                """ p から q への引継ぎ"""
                parent[q] = p
                p, t = q, t + 1
            else:
                """ p から進める点がもう無い時の点 p における処理 """
                if p == start and t == time:
                    break
                p, t = parent[p], t-1
                """ 点 p から親ノードに戻ってきた時の親ノードにおける処理 """
        return -1

    def dfs2(self, info, start = 0, goal = -1, time = 0, save = False):
        """
        :param info: 各頂点の付加情報 = [v1,v2,v3,...,vN]
        :param start: スタート地点
        :param goal: ゴール地点
        :param save: True = 前回の探索結果を保持する
        :return: 伝搬後のinfo
        """
 
        if not save:
            self.parent = [-1] * self.n
        if self.destroy:
            edge2 = self.edges
        else:
            edge2 = marshal.loads(marshal.dumps(self.edges))
 
        p,t=start,time
        self.parent[p]=-2
        while True:
            if edge2[p]:
                q=edge2[p].pop()
                if q==self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                if self.parent[q]!=-1:
                    """ サイクルで同一点を訪れた時の処理 """
                    continue
                if q==goal:
                    """ ゴール時の処理"""
                    self.parent[q]=p
                    return t+1
                """ p から q への引継ぎ"""
                info[q]+=info[p]
                self.parent[q]=p
                p,t=q,t+1
            else:
                """ p から進める点がもう無い時の点 p における処理 """
                if p==start and t==time:
                    break
                p,t=self.parent[p],t-1
                """ 点 p から親ノードに戻ってきた時の親ノードにおける処理 """
        return info
 
    def cycle_detector(self, start = 0, time = 0, save = False):
        """
        :param p: スタート地点
        :param save: True = 前回の探索結果を保持する(遅い)
        :return: サイクルリストを返す。存在しない場合は []
        """
        if not save:
            self.parent = [-1] * self.n
            self.finished=[0]*self.n
        if self.destroy:
            edges2 = self.edges
        else:
            edges2 = [self.edges[p].copy() for p in range(self.n)]

        p, t = start, time
        self.parent[p] = -2
        history = [p]
        cycle = []
        while True:
            if edges2[p]:
                q = edges2[p].pop()
                if q == self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                if self.parent[q] != -1:
                    """ サイクルで同一点を訪れた時の処理 """
                    if not self.finished[q] and not cycle:
                        cycle_start=history.index(q)
                        if save==False:
                            return history[cycle_start:]
                        else:
                            cycle = history[cycle_start:]
                    continue
                """ p から q への引継ぎ"""
                history.append(q)
                self.parent[q] = p
                p, t = q, t + 1
            else:
                """ p から進める点がもう無い時の点 p における処理 """
                self.finished[p]=1
                history.pop()
                if p == start and t == time:
                    break
                p, t = self.parent[p], t-1
                """ 点 p から親ノードに戻ってきた時の親ノードにおける処理 """

        return cycle

    def all_cycles(self, start = 0, time = 0, save = False):
        if not save:
            self.parent = [-1] * self.n
            self.finished=[0]*self.n
        if self.destroy:
            edges2 = self.edges
        else:
            edges2 = [self.edges[p].copy() for p in range(self.n)]

        p, t = start, time
        self.parent[p] = -2
        history = []
        res = []
        while True:
            history.append(p)
            if edges2[p]:
                q = edges2[p].pop()
                if q == self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                if self.parent[q] != -1:
                    """ サイクルで同一点を訪れた時の処理 """
                    if not self.finished[q]:
                        cycle_start=history.index(q)
                        res.append(history[cycle_start:])
                    continue
                """ p から q への引継ぎ"""
                self.parent[q] = p
                p, t = q, t + 1
            else:
                """ p から進める点がもう無い時の点 p における処理 """
                self.finished[p] = 1
                history.pop()
                if p == start and t == time:
                    break
                p, t = self.parent[p], t-1
                """ 点 p から親ノードに戻ってきた時の親ノードにおける処理 """
        return res

    def tree_counter(self, detail=False):
        """
        :param detail: True = サイクルのリストを返す
        :return: 木(閉路を含まない)の個数を返す
        """
        self.parent = [-1] * self.n
        self.finished=[0]*self.n
        connection_number = 0
        cycle_list = []

        for p in range(self.n):
            if self.parent[p] == -1:
                connection_number += 1
                cycle = self.cycle_detector(p, save = True)
                if cycle:
                    cycle_list.append(cycle)
        if not detail:
            return connection_number - len(cycle_list)
        else:
            return cycle_list

    def path_detector(self, start = 0, time = 0, save = False):
        """
        :param p: スタート地点
        :param save: True = 前回の探索結果を保持する
        :return: 各点までの距離と何番目に発見したかを返す
        """

        if not save:
            self.parent = [-1] * self.n

        edges2= []
        for i in range(self.n):
            edges2.append(sorted(self.edges[i], reverse = True))

        p, t = start, time
        self.parent[p] = -2
        full_path = [(p, t)]
        while True:
            if edges2[p]:
                q = edges2[p].pop()
                if q == self.parent[p] and not self.directed:
                    """ 逆流した時の処理 """
                    continue
                if self.parent[q] != -1:
                    """ サイクルで同一点を訪れた時の処理 """
                    continue
                """ p から q への引継ぎ"""
                self.parent[q] = p
                p, t = q, t + 1
                full_path.append((p, t))
            else:
                """ p から進める点がもう無い時の点 p における処理 """
                if p == start and t == time:
                    break
                p, t = self.parent[p], t-1
                """ 点 p から親ノードに戻ってきた時の親ノードにおける処理 """
                full_path.append((p, t))
        return full_path

    def path_list(self):
        """
        :return: 探索経路を返す。
        """
        self.parent = [-1] * self.n
        res = []
        for p in range(self.n):
            if self.parent[p] == -1:
                res.append(self.path_detector(p, time = 1, save = True))
        return res

    def path_restoring(self, start, goal):
        res = []
        while goal != start:
            res.append(goal)
            goal = self.parent[goal]
            if goal < 0: return -1
        res.append(start)
        return res[::-1]

    def minimum_cycle(self):
        """
        無向グラフ用 O(V(V+E))
        """
        res = []
        for start in range(self.n):
            flg = 0
            self.parent = [-1] * self.n
            p, t = start, 0
            self.parent[p] = -2
            next_set = deque()
            starts = set()
            for q in self.edges[p]:
                next_set.append((q, t))
                self.parent[q] = p
                starts.add(q)
            while next_set and flg == 0:
                p, t = next_set.popleft()
                for q in self.edges[p]:
                    if q == self.parent[p] and not self.directed:
                        """ 逆流した時の処理 """
                        continue
                    if self.parent[q]!=-1:
                        """ サイクル時の処理 """
                        if q in starts:
                            cycle = [q]
                            r = p
                            while r not in starts:
                                cycle.append(r)
                                r = self.parent[r]
                                if r < 0: return -1
                            cycle.append(r)
                            cycle.append(start)
                            res.append(cycle[::-1])
                            flg = 1
                            break
                        continue
                    self.parent[q]=p
                    next_set.append((q, t + 1))
        return res

    def minimum_cycle_directed(self):
        """
        有向グラフ用 O(V(V+E))
        """
        res = INF = 10**18
        dist=[]
        history=[]
        for i in range(self.n):
            dist.append(self.distance_list(start = i, save = False, INF = INF)[:])
            history.append(self.parent[:])
        s, g = None, None
        for i in range(self.n):
            for j in self.edges[i]:
                if dist[j][i] + 1 < res:
                    res = dist[j][i] + 1
                    s, g = i, j
        if res >= INF:
            return []
        else:
            self.parent=history[g]
            return self.path_restoring(g, s)

    def path_restoring(self, start, goal):
        res = []
        while goal != start:
            res.append(goal)
            goal = self.parent[goal]
            if goal < 0: return -1
        res.append(start)
        return res[::-1]
    
    def draw(self):
        """
        :return: グラフを可視化
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for x in range(self.n):
            for y in self.edges[x]:
                G.add_edge(x, y)

        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, connectionstyle='arc3, rad = 0.1')
        plt.axis("off")
        plt.show()
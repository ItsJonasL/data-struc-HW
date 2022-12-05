# A naive implementation of union find which can lead to long
# branches
class UnionFind:
    def __init__(self, N):
        self._parent = list(range(N))
    
    def root(self, i):
        """
        Follow parent pointers until reaching a root
        Parameters
        ----------
        i: int
            The starting node 
        
        Returns
        -------
        The root node of i
        """
        while self._parent[i] != i:
            i = self._parent[i]
        return i
    
    def find(self, i, j):
        """
        Return true if i and j are in the same component, or
        false otherwise
        Parameters
        ----------
        i: int
            Index of first element
        j: int
            Index of second element
        """
        return self.root(i) == self.root(j)
    
    def union(self, i, j):
        """
        Merge the two sets containing i and j, or do nothing if they're
        in the same set
        Parameters
        ----------
        i: int
            Index of first element
        j: int
            Index of second element
        """
        root_i = self.root(i)
        root_j = self.root(j)
        if root_i != root_j:
            self._parent[root_j] = root_i
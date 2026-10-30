graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': [],
    'D': ['G'],
    'E': [],
    'F': ['H', 'I'],
    'G': [],
    'H': [],
    'I': []
}


def dfs(graph, node, visited):
    if node not in visited:
        print(node, end=" ")
        visited.add(node)

        for neighbour in graph[node]:
            dfs(graph, neighbour, visited)


visited = set()

print("DFS Traversal:")
dfs(graph, 'A', visited)
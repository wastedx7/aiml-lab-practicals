import heapq

# Manhattan Distance Heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))

    g_cost = {start: 0}
    parent = {}
    visited = set()

    while open_list:
        f, current = heapq.heappop(open_list)

        if current == goal:
            path = []

            while current in parent:
                path.append(current)
                current = parent[current]

            path.append(start)
            path.reverse()

            print("Shortest Path:")
            print(path)
            return

        if current in visited:
            continue

        visited.add(current)

        x, y = current

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = x + dx
            ny = y + dy

            if (0 <= nx < len(grid) and
                0 <= ny < len(grid[0]) and
                grid[nx][ny] == 1):

                neighbor = (nx, ny)
                new_g = g_cost[current] + 1

                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g

                    f = new_g + heuristic(neighbor, goal)
                    parent[neighbor] = current

                    heapq.heappush(open_list, (f, neighbor))

    print("No Path Found")


# New Grid (1 = Walkable, 0 = Obstacle)
grid = [
    [1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1]
]

start = (0, 0)
goal = (4, 4)

astar(grid, start, goal)
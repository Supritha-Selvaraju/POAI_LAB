def minimax(depth, node_index, is_maximizing, scores, height):
    if depth == height:  # Terminal node
        return scores[node_index]

    if is_maximizing:
        return max(minimax(depth + 1, node_index * 2, False, scores, height),
                   minimax(depth + 1, node_index * 2 + 1, False, scores, height))
    else:
        return min(minimax(depth + 1, node_index * 2, True, scores, height),
                   minimax(depth + 1, node_index * 2 + 1, True, scores, height))

# Example Usage
scores = [3, 5, 6, 9, 1, 2, 0, -1]  # Example leaf node values
tree_height = 3  # Height of the game tree
optimal_value = minimax(0, 0, True, scores, tree_height)
print(f"The optimal value is: {optimal_value}")

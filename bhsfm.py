"""
Crowd Leader Identification Module
====================================
Part of the R-MMAF BHSFM (Behavioural Heterogeneity Social Force Model)

Identifies the emergent "leader" in a crowd — the agent furthest
in the direction of collective motion. Used to model herding behaviour
during panic evacuation scenarios.

Author: Sanchit Agarwal
"""

import numpy as np


def identify_leader(positions: np.ndarray, velocities: np.ndarray):
    """
    Identifies the crowd leader using velocity projection.

    Strategy:
    1. Compute mean group velocity vector (collective motion direction)
    2. Project each agent's position onto that direction
    3. The agent with the highest projection is the "leader"

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        (x, y) positions of N crowd agents
    velocities : np.ndarray, shape (N, 2)
        (vx, vy) velocity vectors of N crowd agents

    Returns
    -------
    int or None
        Index of the leader agent, or None if the crowd is stationary
    """
    # Step 1: Average direction the crowd is moving
    group_velocity = np.mean(velocities, axis=0)
    norm = np.linalg.norm(group_velocity)

    if norm == 0:
        return None   # Crowd is stationary — no leader

    group_direction = group_velocity / norm

    # Step 2: Project each position onto the group direction
    projections = np.dot(positions, group_direction)

    # Step 3: Leader = furthest agent in the direction of motion
    return int(np.argmax(projections))


# ----- Quick Demo -----
if __name__ == "__main__":
    np.random.seed(42)
    n_agents = 10

    # Simulate agents moving roughly rightward
    positions  = np.random.rand(n_agents, 2) * 100
    velocities = np.random.rand(n_agents, 2) * 2  # Mostly positive x-direction

    leader = identify_leader(positions, velocities)
    print(f"Leader agent index: {leader}")
    print(f"Leader position:    {positions[leader]}")
    print(f"Leader velocity:    {velocities[leader]}")

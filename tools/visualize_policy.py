"""
Visualize a trained PPO policy playing against a random opponent.

Loads a saved model and renders the game using pygame.
"""

import sys
import time
import torch
import pygame
import numpy as np

from generals.core.action import compute_valid_move_mask
from generals.core import game
from generals.core.game import create_initial_state
from generals.gui import GUI
from generals.gui.properties import GuiMode
from generals.core.rendering import NpGameAdapter

from generals.agents.ppo.network import PolicyValueNetwork, obs_to_array


def random_action(obs):
    """Random valid action."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for d in range(4):
                if mask[i, j, d]:
                    valid.append((i, j, d))
    if not valid:
        return torch.tensor([1, 0, 0, 0, 0], dtype=torch.int32)  # pass
    idx = torch.randint(0, len(valid), (1,)).item()
    row, col, direction = valid[idx]
    is_half = torch.randint(0, 2, (1,)).item()
    return torch.tensor([0, row, col, direction, is_half], dtype=torch.int32)


def load_model(model_path: str, grid_size: int = 4, device: str = "cpu"):
    """Load a trained PPO model from file."""
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create network
    network = PolicyValueNetwork(grid_size=grid_size)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.to(device)
    network.eval()
    
    return network


def main():
    # Parse command line arguments
    model_path = sys.argv[1] if len(sys.argv) > 1 else "pytorch_ppo_model.pth"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}")
    print(f"Rendering at {fps} FPS")
    print(f"Device: {device}")
    print()
    
    # Load the trained model
    network = load_model(model_path, grid_size=4, device=device)
    
    # Initialize game state
    grid = torch.zeros((4, 4), dtype=torch.int32)
    positions = torch.randperm(16)[:2]
    pos_a = (positions[0] // 4, positions[0] % 4)
    pos_b = (positions[1] // 4, positions[1] % 4)
    grid[pos_a] = 1
    grid[pos_b] = 2
    state = create_initial_state(np.array(grid.numpy()))
    
    # Agent names
    agents = ["PPO Agent", "Random"]
    
    # Create game adapter for rendering
    info = game.get_info(state)
    game_adapter = NpGameAdapter(state, agents, info)
    
    # Create agent data for GUI
    agent_data = {
        "PPO Agent": {"color": (255, 0, 0)},  # Red
        "Random": {"color": (0, 0, 255)},      # Blue
    }
    
    # Initialize GUI
    gui = GUI(game_adapter, agent_data, mode=GuiMode.TRAIN, speed_multiplier=1.0)
    
    print("Starting game visualization...")
    print("Controls:")
    print("  - Close window to exit")
    print()
    
    # Game loop
    step_count = 0
    max_steps = 500
    clock = pygame.time.Clock()
    
    while step_count < max_steps:
        # Handle pygame events
        command = gui.tick(fps)
        if command.quit:
            break
        
        # Check if game is done
        info = game.get_info(state)
        if info.is_done:
            winner_idx = int(info.winner)
            winner_name = agents[winner_idx] if winner_idx >= 0 else "Draw"
            print(f"\nGame over! Winner: {winner_name}")
            print(f"Total steps: {step_count}")
            
            # Wait a bit before resetting
            time.sleep(2)
            
            # Reset game
            grid = torch.zeros((4, 4), dtype=torch.int32)
            positions = torch.randperm(16)[:2]
            pos_a = (positions[0] // 4, positions[0] % 4)
            pos_b = (positions[1] // 4, positions[1] % 4)
            grid[pos_a] = 1
            grid[pos_b] = 2
            state = create_initial_state(np.array(grid.numpy()))
            info = game.get_info(state)
            game_adapter.update_from_state(state, info)
            step_count = 0
            print("Starting new game...")
            continue
        
        # Get observations
        obs_p0 = game.get_observation(state, 0)
        obs_p1 = game.get_observation(state, 1)
        
        # PPO agent action (player 0)
        obs_arr = obs_to_array(obs_p0).unsqueeze(0).to(device)
        mask = torch.tensor(compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_p0, value, logprob, entropy = network(obs_arr, mask)
        action_p0 = action_p0.squeeze(0)
        
        # Random agent action (player 1)
        action_p1 = random_action(obs_p1)
        
        # Step the game
        actions = torch.stack([action_p0, action_p1], dim=0)
        new_state, new_info = game.step(state, np.array(actions.cpu().numpy()))
        
        # Update state
        state = new_state
        info = new_info
        
        # Update GUI adapter
        game_adapter.update_from_state(state, info)
        
        step_count += 1
        
        # Control frame rate
        clock.tick(fps)
    
    gui.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()


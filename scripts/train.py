from agents.ppo_agent import PPOAgent
from infrastructure.replay_buffer import ReplayBuffer
from envs.choko_env import Choko_Env
import torch
from .play import agent_v_agent
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np
import tqdm
import os

BATCH_SIZE = 128
ROLLOUT_LENGTH = 4096
HIDDEN_DIM = 64
NUM_EPOCHS = 6
NUM_ITERATIONS = 2000
NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
C1 = 0.5
C2 = 0.01
EPS_CLIP = 0.2
NEG_INF = -1e10

# TODO: add a config file


def run_training_loop():
    writer = SummaryWriter(log_dir = "logs/run_6")
    global_step = 0

    ppo_agent = PPOAgent(num_actions = NUM_ACTIONS, hidden_dim = HIDDEN_DIM)
    ppo_agent_frozen = PPOAgent(num_actions = NUM_ACTIONS, hidden_dim = HIDDEN_DIM)
    ppo_agent_frozen.load_state_dict(ppo_agent.state_dict())
    ppo_agent_frozen.eval()
    ppo_agent_frozen.switch_to_cpu()
    for i in range(NUM_ITERATIONS):
        ppo_agent.eval()
        ppo_agent.switch_to_cpu()
        buffer = ReplayBuffer(gamma = 0.99, lam = 0.95, ppo_agent = ppo_agent, max_size = ROLLOUT_LENGTH)
        dataloader = buffer.make_dataloader(
            batch_size = BATCH_SIZE, 
            shuffle = True, 
            num_workers = 2
            )
        ppo_agent.switch_to_device()
        ppo_agent.train()
        pbar = tqdm.tqdm(total = NUM_EPOCHS, desc = "Training")
        for _ in range(NUM_EPOCHS):
            total_loss = 0.0
            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy_loss = 0.0
            total_clip_frac = 0.0
            total_returns = 0.0
            for batch in dataloader:
                obs, actions, masks, old_logps, advantages, returns = batch
                
                obs = obs.to(ppo_agent.device)
                actions = actions.to(ppo_agent.device)
                old_logps = old_logps.to(ppo_agent.device)
                masks = masks.to(ppo_agent.device)
                advantages = advantages.to(ppo_agent.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns = returns.to(ppo_agent.device)
                total_returns += returns.mean().item()

                # computing the actor loss
                action_logits, values = ppo_agent(obs)
                masked_logits = action_logits + (1 - masks) * NEG_INF
                probs = torch.softmax(masked_logits, dim = -1)
                all_log_probs = torch.log_softmax(masked_logits, dim = -1)
                log_probs = all_log_probs[torch.arange(0, actions.shape[0]), actions]
                
                ratios = torch.exp(log_probs - old_logps)
                clipped_ratios = torch.clamp(ratios, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)

                surrogate_1 = ratios * advantages
                surrogate_2 = clipped_ratios * advantages

                actor_loss = -torch.mean(torch.min(surrogate_1, surrogate_2))
                
                # computing the critic loss
                values = values.squeeze(1)
                critic_loss = torch.mean((returns - values)**2)

                # compute entropy loss
                dist = Categorical(probs)
                entropy_loss = -torch.mean(dist.entropy())
            
                # combining the losses
                loss = actor_loss + C1 * critic_loss + C2 * entropy_loss
                ppo_agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ppo_agent.parameters(), 0.5)
                ppo_agent.optimizer.step()

                # logging
                global_step += 1
                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                writer.add_scalar("Loss/clip_frac", (ratios > 1.0 + EPS_CLIP).float().mean(), global_step)

            avg_loss = total_loss / len(dataloader)
            avg_actor_loss = total_actor_loss / len(dataloader)
            avg_critic_loss = total_critic_loss / len(dataloader)
            avg_entropy_loss = total_entropy_loss / len(dataloader)
            avg_returns = total_returns / len(dataloader)

            writer.add_scalar("Loss/total_loss", avg_loss, global_step)
            writer.add_scalar("Loss/actor_loss", avg_actor_loss, global_step)
            writer.add_scalar("Loss/critic_loss", avg_critic_loss, global_step)
            writer.add_scalar("Loss/entropy_loss", avg_entropy_loss, global_step)
            writer.add_scalar("Returns", avg_returns, global_step)            
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            pbar.update(1)
        
        if (i % 10 == 0):
            
            # evaluate the agent against the frozen agent:
            ppo_agent.eval()
            ppo_agent.switch_to_cpu()
            total_wins = 0
            total_losses = 0
            total_draws = 0
            for j in range(100):
                winner = agent_v_agent(ppo_agent, ppo_agent_frozen)
                if winner == 1:
                    total_wins += 1
                elif winner == 2:
                    total_losses += 1
                else:
                    total_draws += 1
            win_rate = total_wins / (total_wins + total_losses + total_draws)
            draw_rate = total_draws / (total_wins + total_losses + total_draws)
            writer.add_scalar("Eval/win_rate", win_rate, global_step)
            writer.add_scalar("Eval/draw_rate", draw_rate, global_step)

            ppo_agent_frozen.load_state_dict(ppo_agent.state_dict())
            ppo_agent.train()
            ppo_agent.switch_to_device()

            save_path = os.path.join("checkpoints", f"ppo_agent_{i}.pth")
            torch.save({
                "model_state_dict": ppo_agent.state_dict(),
                "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
                "loss": loss.item()
            }, save_path)
            print(f"Iteration {i + 1}/{NUM_ITERATIONS} completed. Model saved to {save_path}")
        
        # TODO, evaluate the agent against frozen agent every 10 iterations, say.

        pbar.close()
    writer.close()


if __name__ == "__main__":
    # run training look
    run_training_loop()

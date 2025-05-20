from agents.ppo_agent import PPOAgent
from infrastructure.replay_buffer import ReplayBuffer
from envs.choko_env import Choko_Env
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np
import tqdm
import os

BATCH_SIZE = 128
ROLLOUT_LENGTH = 4096
HIDDEN_DIM = 64
NUM_EPOCHS = 20
NUM_ITERATIONS = 2000
NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
C1 = 0.5
C2 = 0.01 
EPS_CLIP = 0.2
NEG_INF = -1e10

# TODO: add a config file
# TODO: what should my epochs and updates be?


def run_training_loop():
    writer = SummaryWriter(log_dir = "logs")
    global_step = 0

    ppo_agent = PPOAgent(num_actions = NUM_ACTIONS, hidden_dim = HIDDEN_DIM)
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
            for batch in dataloader:
                obs, actions, masks, old_logps, advantages, returns = batch
                
                obs = obs.to(ppo_agent.device)
                actions = actions.to(ppo_agent.device)
                old_logps = old_logps.to(ppo_agent.device)
                masks = masks.to(ppo_agent.device)
                advantages = advantages.to(ppo_agent.device)
                returns = returns.to(ppo_agent.device)

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
                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar("Loss/actor", actor_loss.item(), global_step)
                writer.add_scalar("Loss/critic", critic_loss.item(), global_step)
                writer.add_scalar("Policy/clip_frac",
                                  (torch.abs(ratios - 1.0) > EPS_CLIP).float().mean().item(),
                                  global_step)
                global_step += 1
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            pbar.update(1)
    
        pbar.close()
    writer.close()

    save_path = os.path.join("checkpoints", "ppo_agent_final.pth")
    torch.save({
        "model_state_dict": ppo_agent.state_dict(),
        "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
        "loss": loss.item()
    }, save_path)

if __name__ == "__main__":
    # run training look
    run_training_loop()

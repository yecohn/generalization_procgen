#world model to learn a representation of the world to help generalization.
#using SPR to train agent
from utils import SequenceReplayBuffer
import hydra
import gym
from ppg_procgen import  Agent, ProcgenEnv
import torch
import numpy as np
from omegaconf import DictConfig
from spr import WorldModel
from torch.optim import Adam
from gym.spaces import Box
import wandb


def load_agent(model_path, envs, device='cuda'):
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path))
    return agent

def make_env(cfg):
    envs = ProcgenEnv(
        num_envs=cfg.num_envs,
        env_name=cfg.env_name,
        num_levels=cfg.num_levels,
        start_level=0,
        distribution_mode="easy",
    )
    if cfg.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/")
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.moveaxis(obs["rgb"],[3,1,2],[1,2,3]))
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=cfg.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    return envs

def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def step(batch, world_model, optim, device='cuda'):
    pred_rew, spr_loss = world_model(batch.observations, batch.actions, batch.rewards)
    nonterminals = 1. - torch.sign(torch.cumsum(batch.dones, 0)).float()
    nonterminals = nonterminals[:world_model.jumps + 1]
    spr_loss = spr_loss*nonterminals
    spr_loss = spr_loss.cpu().mean()
    if len(pred_rew) > 0:
        pred_rew = torch.stack(pred_rew, 0)
        with torch.no_grad():
            reward_target = to_categorical(batch.rewards[:world_model.jumps+1].flatten().to(device), limit=1).view(*pred_rew.shape)
        reward_loss = -torch.sum(reward_target * pred_rew, 2).mean(0).cpu()
        reward_loss = reward_loss.mean()
        optim.zero_grad()
    loss = reward_loss + spr_loss
    loss.backward()
    optim.step()
    return loss.item(), spr_loss.item(), reward_loss.item()
    
        

def interact_with_env(agent, envs, state, rb):
    state_v = torch.from_numpy(state).float().cuda()
    action, _, _, _ = agent.get_action_and_value(state_v)
    action = action.cpu().numpy()
    next_state, reward, done, _ = envs.step(action)
    rb.add(state, next_state, action, reward, done, _)
    state = next_state
    state_v = torch.from_numpy(state).float().cuda()
    if done: 
        state = envs.reset()
        state_v = torch.from_numpy(state).float().cuda()
    return state


def wramup(agent,envs, rb, cfg, initial_state, buffer_size=5e6):
    state = initial_state
    while rb.size() < buffer_size:
        state = interact_with_env(agent, envs, state, rb)
    return state
    
    
    
@hydra.main(config_path="cfg", config_name="config")
def main(cfg:DictConfig):
    env = make_env(cfg)
    print('training for num_levels:', cfg.num_levels, 'env_name:', cfg.env_name)
    #get buffer to collect data
    env_shape = env.observation_space['rgb'].shape
    image_shape = 1, env_shape[2], env_shape[0], env_shape[1]
    observation_space = Box(0, 255, image_shape, dtype=np.uint8)
    env.observation_space = observation_space
    rb = SequenceReplayBuffer(
            buffer_size=cfg.buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            seq_len=cfg.jumps + 1,
            device='cuda',
            n_envs=1)
    #load agent
    del env_shape
    world_model = WorldModel(image_shape=image_shape, output_size=env.action_space.n, **cfg).to('cuda')
    agent = load_agent(cfg.agent_path, env)
    state = env.reset()
    state = wramup(agent, env, rb, cfg, state, buffer_size=1e3)
    batch = rb.sample(batch_size=cfg.batch_size)    
    optimizer = Adam(world_model.parameters(), lr=cfg.lr)
    if cfg.wandb:
        wandb.init(project='spr', config=cfg)
    #training world model via spr and agent
    for i in range(cfg.num_updates):
        batch = rb.sample(batch_size=cfg.batch_size)
        loss, spr_loss, rew_loss = step(batch, world_model, optimizer)
        print(loss, spr_loss, rew_loss)
        if cfg.wandb:
            wandb.log({'loss':loss, 'spr_loss':spr_loss, 'rew_loss':rew_loss}, step=i)
        state = interact_with_env(agent, env, state, rb)
        print(f'loss:{loss}, spr_loss:{spr_loss}, rew_loss:{rew_loss}')
        if i % cfg.save_frequency == 0:
            torch.save(world_model.state_dict(), cfg.world_model_path)
    
#TODO: address the buffer issue to give you also future episodes... 


if __name__ == "__main__": 
    main()
import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_legal_mask=None):
        """
        Saves a transition. Here states are meant to be already processed as tensors,
        or we can process them inside. Assuming they are flat numpy arrays or torch tensors.
        If they are numpy arrays, we can handle it easily.
        """
        if next_legal_mask is None:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.append((state, action, reward, next_state, done, next_legal_mask))
    
    def sample(self, batch_size, device='cpu'):
        batch = random.sample(self.buffer, batch_size)
        has_masks = len(batch[0]) == 6
        if has_masks:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_legal_mask_batch = zip(*batch)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # If they are already tensors:
        if isinstance(state_batch[0], torch.Tensor):
            state_batch = torch.stack(state_batch).to(device)
            next_state_batch = torch.stack(next_state_batch).to(device)
        else:
            state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
            next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
            
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        
        if not has_masks:
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch

        if isinstance(next_legal_mask_batch[0], torch.Tensor):
            next_legal_mask_batch = torch.stack(next_legal_mask_batch).to(device)
        else:
            next_legal_mask_batch = torch.BoolTensor(np.array(next_legal_mask_batch)).to(device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_legal_mask_batch

    def __len__(self):
        return len(self.buffer)

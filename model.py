import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
      super().__init__()
      self.linear1=nn.Linear(input_size,hidden_size)
      self.linear2=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x
    def save(self,file_name='model.pth'):
        model_folder_path='./model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name=os.path.join(model_folder_path,file_name)  
        torch.save(self.state_dict(),file_name)  

class QTrainer:
  def __init__(self,model,lr,gamma):
    self.lr=lr
    self.gamma=gamma
    self.model=model
    self.optimizer=optim.Adam(model.parameters(),lr=self.lr)
    self.criterion=nn.MSELoss()

  def train_step(self, state, action, reward, next_state, done):
    """
    Performs a single step of training for the Q-network using the given batch of experience tuples.

    Args:
        state: Current state(s), shape (batch_size, state_dim) or (state_dim,)
        action: Action(s) taken, shape (batch_size, action_dim) or (action_dim,)
        reward: Reward(s) received, shape (batch_size,) or scalar
        next_state: Next state(s) after taking action, shape (batch_size, state_dim) or (state_dim,)
        done: Boolean(s) indicating if the episode ended, shape (batch_size,) or bool

    This function implements the Q-learning update rule:
        Q_new = reward + gamma * max(Q(next_state))   (if not done)
        Q_new = reward                               (if done)
    The model is trained to minimize the mean squared error between its predictions and these Q_new targets.
    """
    # Convert inputs to torch tensors of appropriate type
    state = torch.tensor(state, dtype=torch.float)
    next_state = torch.tensor(next_state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)

    # If the input is a single sample (not a batch), add a batch dimension
    if len(state.shape) == 1:
      # If the input tensors represent a single sample (i.e., are 1-dimensional),
      # add an extra batch dimension at the front to make them 2-dimensional.
      # This ensures compatibility with the model, which expects batched input.
      state = torch.unsqueeze(state, 0)
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      done = (done,)

    # 1. Get predicted Q values for current state from the model
    pred = self.model(state)

    # 2. Clone the predictions to create the target tensor
    target = pred.clone()

    # 3. For each sample in the batch, update the target Q value for the action taken
    for idx in range(len(done)):
      Q_new = reward[idx]
      # If the episode is not done, add the discounted max Q value for the next state
      if not done[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
      # Find the index of the action taken (for one-hot or index representation)
      # If action is one-hot, use argmax; if it's an index, use directly
      if len(action.shape) > 1 and action.shape[1] > 1:
        action_idx = torch.argmax(action[idx]).item()
      else:
        action_idx = action[idx].item()
      # Update the target for the action taken
      target[idx][action_idx] = Q_new

    # 4. Zero the gradients, compute the loss, and update the model
    self.optimizer.zero_grad()
    loss = self.criterion(target, pred)
    loss.backward()
    self.optimizer.step()
    
  #2:Q_new=r+y*max(next_predicted Q value)->only do this if not done
  # pred.clone()
  # preds[argmax(action)]=Q_new
       
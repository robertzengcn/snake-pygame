import torch
import random
import numpy as np
from game import SnakeGameAI,Direction,Point
from collections import deque
from model import Linear_QNet,QTrainer
from helper import plot


MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:
    def __init__(self):
       self.n_games=0
       self.epsilon=0 # randomness
       self.gamma=0.9 #discount rate
       self.memory=deque(maxlen=MAX_MEMORY) #popleft()
        #Todo:model,trainer
       self.model=Linear_QNet(11,256,3)
       self.trainer=QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,game):
        head=game.snake[0]
        point_l=Point(head.x-20,head.y)
        point_r=Point(head.x+20,head.y)
        point_u=Point(head.x,head.y-20)
        point_d=Point(head.x,head.y+20)

        dir_l=game.direction==Direction.LEFT
        dir_r=game.direction==Direction.RIGHT
        dir_u=game.direction==Direction.UP
        dir_d=game.direction==Direction.DOWN

        # The state list encodes the current situation of the snake and the environment as a set of 11 boolean features.
        # These features are used as input to the neural network for decision making.
        state = [
            # 1. Danger straight: Is there a collision if the snake keeps moving in its current direction?
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # 2. Danger right: Is there a collision if the snake turns right from its current direction?
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # 3. Danger left: Is there a collision if the snake turns left from its current direction?
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)),

            # 4-7. Current move direction: One-hot encoding for [left, right, up, down]
            dir_l,  # Moving left
            dir_r,  # Moving right
            dir_u,  # Moving up
            dir_d,  # Moving down

            # 8-11. Food location relative to the snake's head:
            game.food.x < game.head.x,  # Food is to the left
            game.food.x > game.head.x,  # Food is to the right
            game.food.y < game.head.y,  # Food is above
            game.food.y > game.head.y   # Food is below
        ]
        return np.array(state,dtype=int)
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))# popleft is MAX_MEMORY is reached
    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            # Randomly select a batch of experiences from memory for training.
            # This helps the model learn from a diverse set of past experiences (experience replay).
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample=self.memory
        states,actions,rewards,next_states,dones=zip(*mini_sample)    
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
    def get_action(self,state):
        #random moves:tradeoff exploration/expolotation
        self.epsilon=80-self.n_games  
        final_move=[0,0,0]
        # If a randomly chosen number between 0 and 200 is less than epsilon,
        # choose a random move (exploration). Otherwise, use the model to predict the best move (exploitation).
        if random.randint(0, 200) < self.epsilon:
            move=random.randint(0,2)
            final_move[move]=1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction=self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move]=1
        return final_move
def train():
    plot_scores=[]
    plot_mean_scores=[]
    total_score=0 
    record=0
    agent=Agent()
    game=SnakeGameAI()
    while True:
        #get old state
        state_old=agent.get_state(game)

        #get move
        final_move=agent.get_action(state_old)

        #perform move and get new state
        reward,done,score=game.play_step(final_move)
        state_new=agent.get_state(game)
        #train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            #train long memory,plot result
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()
            if score>record:
                record=score
                agent.model.save()
            print('Game',agent.n_games,'Score',score,'Record:',record)
            #plot
            plot_scores.append(score)
            total_score+=score
            mean_score=total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
                           


if __name__ == "__main__":
   train() 
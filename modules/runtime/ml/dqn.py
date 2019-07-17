import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, inputs, outputs, hidden_layers, neurons):
        super(Net, self).__init__()
        layers = []
        self.first = nn.Linear(inputs, neurons)
        # self.first.weight.data.normal_(0, 0.1)  

        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons, neurons))

        # for layer in layers:
            # layer.weight.data.normal_(0, 0.1)  

        self.last = nn.Linear(neurons, outputs)
        # self.last.weight.data.normal_(0, 0.1)  

        self.layers = layers

    def forward(self, x, dropout):
        x = self.first(x)
        x = F.dropout(x, p=dropout, training=True)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=dropout, training=True)
            x = F.relu(x)

        return self.last(x)


class DQN(object):
    def __init__(self, state_size, num_actions, batch_size, epsilon, gamma, target_iter, memory_size):
        self.state_size = state_size
        self.num_actions = num_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_iter = target_iter

        self.eval_net = Net(state_size, num_actions, 2, 32)
        self.target_net = Net(state_size, num_actions, 2, 32)

        self.learn_step_counter = 0 
        self.memory_counter = 0 
        self.memory = np.zeros((memory_size, state_size * 2 + 2))     
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-2)
        self.loss_func = nn.MSELoss()

    def save(self, filename):
        torch.save(self.eval_net, filename)

    def load(self, filename):
        self.eval_net = torch.load(filename)
        self.target_net = torch.load(filename)

    def sample_uncertainty(self, x, num_samples):
        x = torch.FloatTensor(x).repeat(num_samples, 1)
        samples = self.eval_net.forward(x, 0.05)
        actions = torch.max(samples, 1)[1].data.numpy()
        return np.argmax(np.bincount(actions)), np.var(actions)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:   
            actions_value = self.eval_net.forward(x, 0)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] 
        else:  
            action = np.random.randint(0, self.num_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_size:])

        q_eval = self.eval_net(b_s, 0).gather(1, b_a)
        q_next = self.target_net(b_s_, 0).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

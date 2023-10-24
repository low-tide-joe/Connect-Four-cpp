from DQN import QNetwork
from DQN import train
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_states/DQN_Model.pth"

current_agent = QNetwork().to(device)
# current_agent.load_state_dict(torch.load(model_path))
current_agent.train()

previous_agent = QNetwork().to(device)
previous_agent.load_state_dict(current_agent.state_dict())
previous_agent.eval()

optimizer = optim.Adam(current_agent.parameters(), lr=0.01)
criterion = nn.MSELoss()

train(episodes=1000, current_agent=current_agent, previous_agent=previous_agent, criterion=criterion, optimizer=optimizer, agent_update_frequency=100)

torch.save(current_agent.state_dict(), model_path)

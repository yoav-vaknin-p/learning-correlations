import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
torch.set_default_dtype(torch.float32)


## ---- Global learning parameters ---- ##
epochs_num = 3
train_batch_size = 64
test_batch_size = 500
eta = 0.01  # learning rate
model_act = F.relu # Activation type for the neural net

# Architecture, node numbers per layer
input_nodes_num = 28 * 28
hidden_nodes_num = 256
out_nodes_num = 10

##--  MNIST loading --##
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
)

train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

# Defining the NeuralNet
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Linear fully connected layers
        self.fc1 = nn.Linear(input_nodes_num, hidden_nodes_num, bias=True)
        self.fc2 = nn.Linear(hidden_nodes_num, hidden_nodes_num, bias=True)
        self.fc3 = nn.Linear(hidden_nodes_num, hidden_nodes_num, bias=True)
        self.fc4 = nn.Linear(hidden_nodes_num, out_nodes_num, bias=True)

        # Data aggregation members
        self.hidden_layer_1_data = tensor([])
        self.hidden_layer_2_data = tensor([])
        self.hidden_layer_3_data = tensor([])
        self.output_layer_data = tensor([])

    def forward(self, x):
        # 2d Matrix --> vector  (per sample)
        x = torch.flatten(x, 1)

        # first layer pass
        out1 = self.fc1(x)
        act1 = model_act(out1)

        # first layer recording
        self.hidden_layer_1_data = torch.cat((self.hidden_layer_1_data, out1), dim=0)

        # second layer pass
        out2 = self.fc2(act1)
        act2 = model_act(out2)

        # second layer recording
        self.hidden_layer_2_data = torch.cat((self.hidden_layer_2_data, out2), dim=0)

        # third layer pass
        out3 = self.fc3(act2)
        act3 = model_act(out3)

        # third layer recording
        self.hidden_layer_3_data = torch.cat((self.hidden_layer_3_data, out3), dim=0)

        output = self.fc4(act3)

        # output layer recording
        self.output_layer_data = torch.cat((self.output_layer_data, output), dim=0)
        return output

# Initiating a model
model = NeuralNetwork()


# Loss function
loss_func = nn.CrossEntropyLoss()
# SGD optimizer with chosen learning rate
optimizer = optim.SGD(model.parameters(), lr=eta)

# data-collecting variables, will hold <IMAGES, NUM_OF_NEURONS, EPOCH> tensor at end of training
layer_1_data, layer_2_data, layer_3_data, output_layer_overall_data = tensor([]),tensor([]), tensor([]), tensor([])


## -- Train loop -- ##

# Loss tracking
running_loss = 0.0

for epoch in range(1):
    # Initializing data-collecting members of the model to empty tensors every epoch
    model.hidden_layer_1_data = tensor([])
    model.hidden_layer_2_data = tensor([])
    model.hidden_layer_3_data = tensor([])
    model.output_layer_data = tensor([])


    for step, (train_x, train_y) in enumerate(train_loader):

        # forward pass #
        y_pred = model(train_x)

        # Backpropagation #
        # No need to keep old gradients from backward() calls
        optimizer.zero_grad()

        # Calculate loss
        loss = loss_func(y_pred, train_y)

        # Calculate gradients
        loss.backward()

        # SGD step
        optimizer.step()

        # Loss logging
        running_loss += loss.item()
        if step % 150 == 0:
            # Average loss of 150 SGD steps
            print("Average loss for last 150 steps : ", running_loss/150)
            running_loss = 0.0


    # -- End of Epoch -- #

    # Updating global data-collectors
    layer_1_data = torch.cat((torch.unsqueeze(model.hidden_layer_1_data, dim=2), layer_1_data), dim=2)
    layer_2_data = torch.cat((torch.unsqueeze(model.hidden_layer_2_data, dim=2), layer_2_data), dim=2)
    layer_3_data = torch.cat((torch.unsqueeze(model.hidden_layer_3_data, dim=2), layer_3_data), dim=2)
    output_layer_overall_data = torch.cat((torch.unsqueeze(model.output_layer_data, dim=2), output_layer_overall_data), dim=2)


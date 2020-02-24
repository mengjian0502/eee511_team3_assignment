from torch import nn
import torch
from torch import tensor
from torch.utils import Dataset, Dataloader
import os

class HappinessDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.features = torch.load('input_features.pt')
        self.score = torch.load('happy_score.pt')
        self.train_dataset, self.train_target = features[:int(0.8 * features.size(0)), :], score[:int(0.8 * features.size(0))]
        self.valid_dataset, self.valid_target = features[int(0.2 * features.size(0)):-1, :], score[int(0.2 * features.size(0)):-1]

    
    def __getitem__(self, index):
        return self.features[index], self.score[index]

    def __len__(self):
        print(train_dataset.size())
        print(train_target.size())
        return self.len
    

dataset = HappinessDataset()

training_loader = train_loader(train_dataset.float(), train_target.float(), batch=64,shuffle=True,num_worker=2)
val_loader = valid_loader(valid_dataset.float(), valid_target.float(), batch=64,shuffle=True,num_worker=2)



class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(6, 1)  # six features in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    for i,data in enumerate(training_loader,0):
        #get the inputs
        inputs, labels = data

        #wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # 1) Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # 2) Compute and print loss
        loss = criterion(y_pred, labels)
        print(f'Epoch: {i} | Loss: {loss.item()} ')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
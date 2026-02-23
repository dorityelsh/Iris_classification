
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import datasets
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

datasets.load_iris()
device = "cpu"

data = datasets.load_iris()

my_df = pd.DataFrame(data.data, columns=data.feature_names)
my_df['target'] = data.target
my_df["target_name"] = my_df["target"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(my_df.head())


class Model(nn.Module):
    #input layer 4 features of the flower
    # hidden layer1 number of nueurons 
    # H2(n) 
    #  output layer s classes of the flower
    def __init__(self, input_size =4 , h1=8, h2 = 9, output_size=3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# set seed to fix randomness so results are reproducible
torch.manual_seed(41)
# Create an instance of the model
model = Model()

#train test split and convert to numpy arrays
x = my_df.drop(columns=['target','target_name']).values
y = my_df['target'].values

# random_state makes the train/test split the same every time
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

#convert to tensors float and long 
#x_train = torch.tensor(x_train, dtype=torch.float32)
# device = "cuda" if torch.cuda.is_available() else "cpu"

#x_train = torch.FloatTensor(x_train)
x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)

y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)


# move data to the same device as the model (GPU or CPU)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

model = model.to(device)


#set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
#choose Adam optimaizer , lr - lerarning rate (if error doesnt go down lower the learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train our model 
# epoch ? - how many times we want to loop through the data
epochs = 100
losses = []




for i in range(epochs):
    optimizer.zero_grad() # zero the gradients before backpropagation
    # go forward and get the predictions
    y_pred = model(x_train)
    # calculate the loss , gonna be high at first and then should go down as we train the model
    loss = criterion(y_pred, y_train) # prediction value vs the y_train value
    #keep track of the loss
    losses.append(loss.item())
    #print the loss every 10 epochs
    if i % 10 == 0:
        print(f"epoch: {i}, loss: {loss.item():.4f}")
    # do some back propagation to calculate the gradients: take the error rate of forward propagation and feed it back
    #thru the network to find tune the wights
    print(loss)
    print(loss.requires_grad)

    
    loss.backward() # backpropagation to calculate gradients
    optimizer.step() # update the weights based on the gradients

# plt.plot(range(epochs), losses)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss Over Time")
# plt.show()

print("done")
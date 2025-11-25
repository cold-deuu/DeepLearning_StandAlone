import torch
import torchvision

from torch import nn
import torch.optim as optim



class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Conv2d(in_channels=1,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True) # output : batch, 32, 28, 28
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2) # output : batch, 32, 14, 14

        self.activate1 = nn.ReLU()
        self.layer2 = nn.Linear(32*14*14, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.activate1(out)
        out = self.pooling1(out)
        out = out.reshape(x.shape[0], -1)
        out = self.layer2(out)
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, data_loader, optimizer, epoch):
    loss_func = nn.CrossEntropyLoss().to(device)

    for i in range(epoch):
        for data, target in data_loader:
            # print(f"data shape : {data.shape}")

            optimizer.zero_grad()
            # mnist_data = data.unsqueeze(1).reshape(data.shape[0], -1).to(device)
            data = data.to(device)
            target = target.to(device)

            # data : batch_size, 784
            y = model(data)

            loss = loss_func(y, target)
            loss.backward()
            optimizer.step()
        print(f"epoch : {i}/{epoch}")

def test(model, data_loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, target in data_loader:

            # mnist_data = data.view(data.shape[0], -1).to(device)
            data = data.to(device)
            target = target.to(device)

            output_logits = model(data)
            output_probs = torch.softmax(output_logits, dim=1)
            _, predicted = torch.max(output_probs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            print(f"Predict: {predicted}, Correct: {target}")

    print(f"Accuracy: {100 * correct / total:.2f}%")

def main():

    dataset = torchvision.datasets.MNIST('./', download=True, transform=torchvision.transforms.ToTensor())
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) - train_size)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 512, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 4, shuffle=True)
    
    model = SimpleCNN().to(device)



    epoch = 15
    learning_rate = 0.001

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # print(model.parameters().keys())
    train(model, train_data_loader, optimizer, epoch)
    test(model, test_data_loader)


if __name__=="__main__":
    main()
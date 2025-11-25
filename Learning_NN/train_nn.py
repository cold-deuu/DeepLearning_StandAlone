import torch
import torchvision

from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, 512)
        self.layer_2 = nn.Linear(512, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.sigmoid(self.layer_1(x))
        # y2 = self.sigmoid(self.layer_2(y1))
        y2 = self.layer_2(y1)
        return y2

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, data_loader, epoch, lr = 1e-3):
    loss_func = nn.CrossEntropyLoss().to(device)

    for i in range(epoch):
        for data, target in data_loader:
            mnist_data = data.unsqueeze(1).reshape(data.shape[0], -1).to(device)
            target = target.to(device)

            # data : batch_size, 784
            y = model(mnist_data)

            loss = loss_func(y, target)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr

        print(f"epoch : {i}/{epoch}")

def test(model, data_loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            mnist_data = data.view(data.shape[0], -1).to(device)
            target = target.to(device)

            output_logits = model(mnist_data)
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
    
    model = SimpleNN(784, 10).to(device)
    
    epoch = 15
    learning_rate = 0.001

    
    # print(model.parameters().keys())
    train(model, train_data_loader, epoch, learning_rate)
    test(model, test_data_loader)


if __name__=="__main__":
    main()
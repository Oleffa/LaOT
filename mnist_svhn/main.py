from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class SVHN(nn.Module):
    def __init__(self):
        super(SVHN, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=5))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2, 2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2, 2))

        self.feat = nn.Sequential()
        self.feat.add_module('c_fc1', nn.Linear(1200, 100))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, return_feature=False):
        if input_data.shape[1] == 1:
            input_data = input_data.repeat(1, 3, 1, 1)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 5 * 5)
        feature = self.feat(feature)
        class_output = self.class_classifier(feature)
        if return_feature:
            return class_output, feature
        else:
            return class_output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def record(model, device, loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            labels.append(target)
            _, f = model(data, return_feature=True)
            features.append(f)
    features = torch.cat(features, 0).cpu().detach().numpy()
    labels = torch.cat(labels, 0).cpu().detach().numpy()
    return features, labels

def main():
    # Training settings
    # Obtain Mnist or SVHN features

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=12, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Which dataset to use, mnist or svhn')
    parser.add_argument('--save-features', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset == 'mnist':
        transform= transforms.Compose([transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                      mean=(0.5),
                                      std=(0.5)
                                 )])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                           transform=transform)

    elif args.dataset == 'svhn':
        transform=transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])
        dataset1 = datasets.SVHN('../data/SVHN', split='train', download=True,
                           transform=transform)
        dataset2 = datasets.SVHN('../data/SVHN', split='test', download=True,
                           transform=transform)
    else:
        print('Error, unknown dataset {}'.format(args.dataset))
        exit()

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = SVHN()

    if args.save_features:
        model.load_state_dict(torch.load('svhn_cnn.pt'))
        features, labels = record(model, device, train_loader)
        np.save(args.dataset + '_features.npy', features)
        np.save(args.dataset + '_labels.npy', labels)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=12, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
        if args.save_model:
            if args.dataset == 'mnist':
                torch.save(model.state_dict(), "mnist_cnn.pt")
            else:
                torch.save(model.state_dict(), "svhn_cnn.pt")

if __name__ == '__main__':
    main()

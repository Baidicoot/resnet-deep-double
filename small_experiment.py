import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time
import pickle

from resnet import make_resnet18k

BATCH_SIZE=128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainset = torchvision.datasets.CIFAR10(root='./raw_datas/cifar10', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./raw_datas/cifar10', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)


lossfn = nn.CrossEntropyLoss(label_smoothing=0.2)

def errfn(logits, target):
    idx, pred = logits.max(1)
    return pred.eq(target).sum()

# inner per-model procedure
def test_with_k(k):
    model = make_resnet18k(k)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) 

    # train loop
    def train():
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        nbatches = 0

        for inputs, targets in trainloader:
            nbatches += 1

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = lossfn(logits, targets)
            err = errfn(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += err.item()
        return train_loss/nbatches, 1-correct/total

    # test loop
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        nbatches = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                nbatches += 1

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = lossfn(logits, targets)
                err = errfn(logits, targets)
                loss.backward()
                optimizer.step()

                test_loss += loss.item()
                total += targets.size(0)
                correct += err.item()
            return test_loss/nbatches, 1-correct/total

    print(f"using {sum(p.numel() for p in model.parameters())*10**-6:.2f}M parameter model")

    results = []

    now = time.time()
    dt = 0
    for epoch in range(0, 500):
        train_loss, train_error = train(epoch)
        test_loss, test_error = test(epoch)
        print(f"epoch: {epoch:04}\n\
            dt: {dt:.4f}s\n\
            train loss: {train_loss:.4f}\n\
            train error: {train_error:.4f}\n\
            test loss: {test_loss:.4f}\n\
            test error: {test_error:.4f}")
        results.append({"train_loss": train_loss, "train_error": train_error, "test_loss": test_loss, "test_error": test_error})
        now_ = time.time()
        dt = now_ - now
        now = now_

    # save results to file
    with open(f'deep_double/resnet/resnet18_{k}', "wb") as f:
        pickle.dump(results, f)

widths = [n for n in range(1,33)]

for i, k in enumerate(widths):
    print(f"training model {i+1}/{len(widths)}, k={k}")
    test_with_k(k)
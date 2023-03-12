import matplotlib.pyplot as plt
import pickle

EPOCH=1499

idx = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]

test_losses=[]
train_losses=[]

for i in idx:
    with open(f"deep_double/resnet_0/resnet18_{i}", "rb") as f:
        d = pickle.load(f)
        test_losses.append(d[EPOCH]["test_loss"])
        train_losses.append(d[EPOCH]["train_loss"])

l=[2**n for n in range(0,7)]

plt.plot(idx, list(zip(test_losses, train_losses)))
plt.xscale("log")
plt.xlabel("Model size (k)")
plt.ylabel("Loss")

plt.gca().set_xticks(l)
plt.gca().set_xticklabels(l)

plt.legend(["Test", "Train"])

plt.show()
import matplotlib.pyplot as plt
import pickle

SIZE=24

test = []
train = []

with open(f"deep_double/resnet_0/resnet18_{SIZE}", "rb") as f:
    d = pickle.load(f)
    for e in d:
        test.append(e["test_loss"])
        train.append(e["train_loss"])

l=[1,10,100,1500]

plt.plot(list(zip(test, train)))
plt.xscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.gca().set_xticklabels(l)
plt.gca().set_xticks(l)

plt.legend(["Test", "Train"])

plt.show()
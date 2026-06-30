from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    '../data',
    train=True,
    download=False,
    transform=transform
)

print(len(dataset))
img, label = dataset[0]
print(img.shape, label)
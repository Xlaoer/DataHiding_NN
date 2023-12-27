from train import *


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True
)




def test_image_reconstruction(net, testloader):
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 3, 32, 32).cpu().data
        save_image(outputs, 'conv_cifar10_reconstruction.png')
        break


if __name__ == '__main__':
    device = get_device()
    print(device)
    net.to(device)
    make_dir()
    train_loss = train(net, trainloader, NUM_EPOCHS)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('conv_ae_cifar10_loss.png')
    test_image_reconstruction(net, testloader)



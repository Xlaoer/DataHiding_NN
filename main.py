from train import *
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import argparse



transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ImageDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.imgs = os.listdir(root_dir)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.imgs[index])
    img = Image.open(img_path).convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
    return img

  def __len__(self):
    return len(self.imgs)

trainset = ImageDataset('train_images',transform=transform)
testset = ImageDataset('test_images',transform=transform)

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
        img = batch
        save_image(img,'test_origin.png')
        img = img.to(device)
        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 3, 256, 256).cpu().data
        save_image(outputs, 'test_reconstruction.png')
        print('PSNR for this test is {}'.format(calculate_psnr('test_origin.png','test_reconstruction.png')))
        break


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(description='Training of nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_parser = subparsers.add_parser('new', help='train & test')
    continue_parser = subparsers.add_parser('continue', help='test a existing model')
    args = parent_parser.parse_args()
    device = get_device()
    print(device)
    net.to(device)
    make_dir()
    if args.command == 'continue':
        net = torch.load("./model/model1.pth")  # 导入模型参数
    else:
        train_loss = train(net, trainloader, NUM_EPOCHS)
        plt.figure()
        plt.plot(train_loss)
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('conv_ae_cifar10_loss.png')
    test_image_reconstruction(net, testloader)
    if args.command != 'continue':
        torch.save(net, "./model/model1.pth")



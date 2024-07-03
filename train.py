import argparse
import torch
import torch.optim as optim
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.utils.data as data
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Udacity Training')

    parser.add_argument('--data_dir', default='./flowers/', help='path to dataset')

    parser.add_argument('--save_dir', default='checkpoint.pth', help='path to save checkpoint')

    parser.add_argument('--arch', default='alexnet', choices=['vgg13', 'vgg16', 'alexnet'], help='model architecture')

    parser.add_argument('--hidden_units', default=2048, type=int, help='hidden units')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')

    parser.add_argument('--epochs', default=5, type=int, help='number of epochs')

    parser.add_argument('--gpu', default=True, type=bool, help='whether to use gpu')

    return parser.parse_args()


def train_data_transform(train_dir):
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    train_data_transforms = data.DataLoader(train_data, batch_size=50, shuffle=True)

    class_to_idx = train_data.class_to_idx

    return train_data_transforms, class_to_idx


def valid_data_transform(train_dir):
    valid_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    return data.DataLoader(datasets.ImageFolder(train_dir, transform=valid_transforms), batch_size=50, shuffle=True)


def validation(device, model, valid_loader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


def train_data(args, device, model, train_data_loader, val_data_loader, class_to_idx):
    model.to(device)

    learning_rate = args.learning_rate
    epochs = args.epochs

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    if epochs is None:
        epochs = 5

    print("Training start.")

    avg_train_loss = 0

    avg_val_loss = 0

    avg_train_accuracy = 0

    print_every = 20

    for e in range(epochs):
        steps = 0

        running_loss = 0

        accuracy = 0

        model.train()

        for images, labels in train_data_loader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)

            train_loss = criterion(outputs, labels)

            train_loss.backward()

            optimizer.step()

            running_loss += train_loss.item()

            if steps % print_every == 0 or steps == len(train_data_loader):
                model.eval()

                with torch.no_grad():
                    val_loss, accuracy = validation(device, model, val_data_loader, criterion)

                    average_train_loss_batch = running_loss / print_every

                    average_val_loss_batch = val_loss / len(val_data_loader)

                    average_train_accuracy_batch = accuracy / len(val_data_loader) * 100

                    print(
                        "Epoch:Batch {}/{}, Completed: {:.2f}%,Training Loss: {:.2f},Valid Loss: {:.2f},Accuracy: {"
                        ":.2f}%".format(
                            e + 1,
                            args.epochs,
                            steps * 100 / len(train_data_loader),
                            average_train_loss_batch,
                            average_train_loss_batch,
                            average_train_accuracy_batch
                        ))

    checkpoint = {
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': args.arch
    }

    if args.save_dir:
        torch.save(checkpoint, args.save_dir)
    else:
        torch.save(checkpoint, 'checkpoint.pth')


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arch == 'vgg13':
        model = models.vgg13()
        input_size = 25088
    elif args.arch == 'vgg16':
        model = models.vgg16()
        input_size = 25088
    else:
        model = models.alexnet()
        input_size = 9216

    data_dir = 'flowers/'
    train_dir = data_dir + 'train'
    val_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    train_data_transforms, class_to_idx = train_data_transform(train_dir)
    val_data_transforms = valid_data_transform(val_dir)
    test_data_transforms = valid_data_transform(test_dir)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)


    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(input_size, args.hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.3)),
            ('fc2', nn.Linear(args.hidden_units, 2048)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )

    model.classifier = classifier

    train_data(args, device, model, train_data_transforms, val_data_transforms, class_to_idx)

    print("Training end.")


if __name__ == '__main__':
    main()

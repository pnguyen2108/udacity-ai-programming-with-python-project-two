import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image as Image
import json


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='./flowers/test/1/image_06752.jpg')

    parser.add_argument('--load_model', type=str, default='./checkpoint.pth')

    parser.add_argument('--top_k', type=int, default=5)

    parser.add_argument('--category_names', type=str, default='')

    parser.add_argument('--gpu', type=str, default='')

    args = parser.parse_args()

    return args


def load_model(path):
    checkpoint = torch.load(path)

    arch = checkpoint['arch']

    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)

    model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False

    return model


def predict(image_path, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    image = Image.open(image_path)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )

    tensor_img = transform(image).unsqueeze(0)

    tensor_img = tensor_img.float().to(device)

    with torch.no_grad():
        output = model.forward(tensor_img)

    probabilities = torch.exp(output)

    return probabilities.topk(topk)


def main():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

    args = parse_arguments()

    model = load_model(args.load_model)

    probabilities = predict(args.image_path, model, args.top_k)

    mapping_value = {val: key for key, val in model.class_to_idx.items()}

    classes = probabilities[1][0].cpu().numpy()

    classes = [mapping_value[item] for item in classes]

    classes = [cat_to_name[str(index)] for index in classes]

    for l in range(args.top_k):
        print("\"{}\" with a probability of: {:.2f}%.".format(classes[l], probabilities[0][0].cpu().numpy()[l] * 100))


if __name__ == '__main__':
    main()

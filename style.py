from PIL import Image
import torch
from torchvision.transforms.v2 import ToImage, ToDtype, Resize, Compose
import torchvision.models as models
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class StyleModel(nn.Module):
    def __init__(self):
        super(StyleModel, self).__init__()
        self.model = models.vgg19(weights='DEFAULT')
        self.model.requires_grad_(False)
        self.model.eval()
        self.indices = [0, 5, 10, 19, 28, 34]
        self.num_layers = len(self.indices) - 1

    def forward(self, x):
        output = []
        i = 0
        for layer in self.model.features:
            x = layer(x)
            if i in self.indices:
                output.append(x.squeeze(0))
        return output


def get_content_loss(content_img, create_img):
    return torch.mean(torch.square(content_img - create_img))


def gram_matrix(image):
    channels = image.shape[0]
    g = image.view(channels, -1)
    return torch.mm(g, g.mT) / g.size(dim=1)


def get_style_loss(style_img, base_gram):
    lmd = [1, 0.5, 0.4, 0.8, 0.4]
    loss = 0
    i = 0
    for style, base in zip(style_img, base_gram):
        gram = gram_matrix(style)
        loss += lmd[i] * torch.mean(torch.square(gram - base))
        i += 1
    return loss


img = Image.open('gray_cat.jpg').convert('RGB')
img_style = Image.open('style_2.jpg').convert('RGB')

transform = Compose([
    ToImage(),
    ToDtype(dtype=torch.float32, scale=True)
])

img_style = transform(img_style).unsqueeze(0)
img = transform(img).unsqueeze(0)
img_create = img.clone()
img_create.requires_grad_(True)

model_style = StyleModel()
base_res = model_style(img)[-1]
style_res = model_style(img_style)
gram_style = [gram_matrix(s) for s in style_res[:model_style.num_layers]]

optimizer = optim.Adam([img_create], lr=0.01)

a, b = 1, 1000
best_img = img_create.clone()
best_loss = -1

for epoch in range(100):
    res = model_style(img_create)

    content_loss = get_content_loss(res[-1], base_res)
    style_loss = get_style_loss(res, gram_style)
    loss = a * content_loss + b * style_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    img_create.data.clamp_(0, 1)

    if loss < best_loss or best_loss < 0:
        best_loss = loss
        best_img = img_create.clone()

    print(f"epoch {epoch}, loss {loss}")


x = best_img.detach().squeeze()
lo, hi = x.amin(), x.amax()
x = (x - lo) / (hi - lo) * 255.0
x = x.permute(1, 2, 0).numpy()

x = np.clip(x, 0, 255).astype(np.uint8)

image = Image.fromarray(x)
image.save('style_cat.jpeg')
plt.imshow(x)
plt.show()

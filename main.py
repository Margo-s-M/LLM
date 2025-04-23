
import torch   #основна бібліотека для глибокого навчання.
from torchvision import transforms  #модуль для попередньої обробки зображень
from torchvision import models #містить готові архітектури моделей (в т.ч. resnet50).
from torch import nn  #модуль для створення нейронних мереж.
import torch.nn.functional as F #функціональний API для активаційF.relu, F.softmax, F.sigmoid, втрат(loss): F.cross_entropy, F.mse_loss, F.nll_loss ,лінійні операції: F.linear, F.conv2d, F.batch_norm nn створює об'єкт (наприклад, модуль), а F — викликає операцію без збереження стану.
import PIL.Image  #для відкриття та обробки зображень.
# 🔧 Клас моделі на основі ResNet50
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Завантаження ResNet50
        resnet= models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        # Замінюємо останній шар на Identity
        resnet.fc = nn.Identity()
        self.resnet = resnet

        # Зберігаємо ResNet як підмережу
        self.linear = nn.Linear (2048, 2)

    def forward(self, x):
        out = self.resnet(x)
        out = self.linear(out)
        return out


# 📦 Функція завантаження моделі
def load_model(weights_path: str) -> MyModel:
    model = MyModel()
    model.load_state_dict(torch.load('data/model_weights.pth', weights_only=True))
    model.eval()
    return model


# 🖼️ Функція обробки зображення
def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = PIL.Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


# 🔍 Функція передбачення
def predict(model: MyModel, image_tensor: torch.Tensor, classes: list[str]) -> str:
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_idx = probabilities.argmax().item()

    print(f"Ймовірності: {probabilities.numpy()}")
    print(f"Передбачено: {classes[predicted_idx]}")
    return classes[predicted_idx]


# 🚀 Основний блок запуску
if __name__ == "__main__":
    model_path = "data/model_weights.pth"
    image_path = "data/000000000023.jpg"
    class_labels = ['bird', 'drone']

    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    result = predict(model, image_tensor, class_labels)
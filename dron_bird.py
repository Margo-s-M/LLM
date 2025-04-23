import torch   # Основна бібліотека для глибокого навчання PyTorch.
from torchvision import transforms  # Модуль для попередньої обробки зображень (зміна розміру, нормалізація тощо).
from torchvision.models import ResNet50_Weights  # Імпорт попередньо натренованих ваг для ResNet50.
from torch import nn  # Модуль для створення нейронних мереж.
import torch.nn.functional as F  # Функціональний API для активацій, втрат і базових операцій.
import PIL.Image  # Для відкриття та обробки зображень.
from torchvision import models  # Імпорт готових архітектур моделей.

# Оголошення власного класу моделі на основі ResNet50.
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()  # Викликаємо конструктор батьківського класу.

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Завантажуємо попередньо натреновану модель ResNet50.
        for param in resnet.parameters():
            param.requires_grad = False  # Заморожуємо ваги усіх шарів, щоб їх не перенавчати.

        resnet.fc = nn.Identity()  # Замінюємо останній повнозв'язний шар на 'пустий', щоб отримати вихід перед класифікацією.
        self.resnet = resnet  # Зберігаємо ResNet як підмережу.

        self.linear = nn.Linear(2048, 2)  # Додаємо власний лінійний шар для класифікації на 2 класи.

    def forward(self, x):  # Метод прямого проходу через модель.
        out = self.resnet(x)  # Пропускаємо зображення через ResNet.
        out = self.linear(out)  # Потім через власний лінійний шар.
        return out  # Повертаємо логіти.


# Функція завантаження моделі з файла ваг.
def load_model(weights_path: str) -> MyModel:
    model = MyModel()  # Ініціалізуємо екземпляр моделі.
    model.load_state_dict(torch.load('data/model_weights.pth', weights_only=True))  # Завантажуємо збережені ваги.
    model.eval()  # Переводимо модель у режим оцінювання (inference mode).
    return model


# Функція для попередньої обробки зображення перед подачею в модель.
def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Змінюємо розмір зображення до 224x224.
        transforms.ToTensor(),  # Перетворюємо зображення в тензор.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Нормалізуємо зображення за середнім та відхиленням ImageNet.
                             std=[0.229, 0.224, 0.225])
    ])
    image = PIL.Image.open(image_path).convert('RGB')  # Відкриваємо зображення та перетворюємо в RGB.
    return transform(image).unsqueeze(0)  # Застосовуємо трансформації і додаємо розмір батчу (1).


# Функція для передбачення класу зображення.
def predict(model: MyModel, image_tensor: torch.Tensor, classes: list[str]) -> str:
    with torch.no_grad():  # Вимикаємо обчислення градієнтів для прискорення та економії памʼяті.
        logits = model(image_tensor)  # Отримуємо логіти від моделі.
        probabilities = F.softmax(logits, dim=1)  # Застосовуємо Softmax для перетворення логітів у ймовірності.
        predicted_idx = probabilities.argmax().item()  # Вибираємо індекс класу з найбільшою ймовірністю.

    print(f"Ймовірності: {probabilities.numpy()}")  # Виводимо ймовірності для всіх класів.
    print(f"Передбачено: {classes[predicted_idx]}")  # Виводимо передбачений клас.
    return classes[predicted_idx]  # Повертаємо передбачене текстове значення класу.


# Основний блок запуску коду.
if __name__ == "__main__":
    model_path = "data/model_weights.pth"  # Шлях до збережених ваг моделі.
    image_path = "data/03.jpg"  # Шлях до зображення, яке потрібно класифікувати.
    class_labels = ['bird', 'drone']  # Назви класів.

    model = load_model(model_path)  # Завантажуємо модель.
    image_tensor = preprocess_image(image_path)  # Проводимо обробку зображення.
    result = predict(model, image_tensor, class_labels)  # Виконуємо передбачення і виводимо результат.


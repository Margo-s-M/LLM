import torch   #основна бібліотека для глибокого навчання.
from torchvision import transforms  #модуль для попередньої обробки зображень
from torchvision import models #містить готові архітектури моделей (в т.ч. resnet50).
from torch import nn  #модуль для створення нейронних мереж.
import torch.nn.functional as F #функціональний API для активаційF.relu, F.softmax, F.sigmoid, втрат(loss): F.cross_entropy, F.mse_loss, F.nll_loss ,лінійні операції: F.linear, F.conv2d, F.batch_norm nn створює об'єкт (наприклад, модуль), а F — викликає операцію без збереження стану.
import PIL.Image  #для відкриття та обробки зображень.

#створюємо свій клас MyModel, який успадковує PyTorch-модель (nn.Module)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # викликає конструктор батьківського класу
#Підключення ResNet50()Завантажуємо попередньо натреновану ResNet50 (на ImageNet).Заморожуємо всі ваги: requires_grad = False — щоб не тренувалися заново
        resnet = models.resnet50(pretrained=True)
        for parameter in resnet.parameters():
            parameter.requires_grad = False

        resnet.fc = nn.Identity()  #Видаляємо останній шар замінюємо на Identity (нічого не змінює, просто передає дані далі).
        self.resnet = resnet  #Додаємо свій класифікатор.Зберігаємо ResNet без останнього шару як self.resnet

        self.linear = nn.Linear(2048, 2)  #Додаємо свій Linear шар (2048 → 2 класи).

    def forward(self, x):
        out = self.resnet(x)   # Витягуємо ознаки
        out = self.linear(out)  # Класифікуємо

        return out

#Завантаження моделі
# Створюємо об'єкт моделі
# Завантажуємо натреновані ваги.
# eval() переводить модель у режим передбачення
# (inference) (відключає dropout, batchnorm в режим тренування тощо)

model= MyModel()
model.load_state_dict(torch.load('data/model_weights.pth', weights_only=True))
model.eval()

# Resize — підганяємо розмір під вимоги ResNet.
# ToTensor — конвертуємо зображення у тензор PyTorch
# Normalize — нормалізуємо пікселі (як було при тренуванні ResNet)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = ['bird', 'drone']  #classes[0] → 'bird'  classes[1] → 'drone'

#Завантаження зображення
img = PIL.Image.open('data/01.jpg')
img.show("title")

#Перетворення зображення
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)


#відключає обчислення градієнтів (швидше та економніше)
#переконуємося, що модель на CPU
#Пропускаємо зображення через модель → отримуємо логіти (не ймовірності).
with torch.no_grad():
    model = model.to('cpu')
    logits = model(img_tensor) #вихідні значення моделі перед застосуванням функції softmax

#softmax перетворює логіти у ймовірності (всі значення від 0 до 1, сума = 1)
proba = F.softmax(logits)

#argmax() → індекс класу з найбільшою ймовірністю
idx = proba.argmax()

print(logits)# необроблені логіти
print(idx)  # індекс передбаченого класу
print(proba) # ймовірності




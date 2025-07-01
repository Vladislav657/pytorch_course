import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


_global_var_text = [
    "Как я отмечал во введении, простейшая НС – персептрон, представляет собой ...",
    "Это классический пример полносвязной сети ...",
    "Каждая связь между нейронами имеет определенный ...",
]

# сюда копируйте класс CharsDataset из предыдущего подвига
class CharsDataset(data.Dataset):
    def __init__(self, prev_chars=10):
        all_text = ''.join(_global_var_text)
        self.alphabet = set(all_text.lower())
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}

        fragments = [[s[i:i+prev_chars+1].lower() for i in range(len(s) - prev_chars)] for s in _global_var_text]
        self.dataset = []
        for fragment in fragments:
            self.dataset.extend(fragment)
        self.count = len(self.alphabet)

    def __getitem__(self, index):
        d = self.dataset[index]
        codes = [self.alpha_to_int[alpha] for alpha in d]
        return torch.eye(self.count)[codes[:-1]], torch.tensor(codes[-1])

    def __len__(self):
        return len(self.dataset)

# здесь объявляйте класс модели нейронной сети
class CharsModel(nn.Module):
    def __init__(self):
        super(CharsModel, self).__init__()
        n_chars = CharsDataset().count
        self.rnn = nn.RNN(input_size=n_chars, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, n_chars)

    def forward(self, x):
        _, h = self.rnn(x)
        # print(h.shape)
        return self.linear(h.squeeze(0))

# сюда копируйте объекты d_train и train_data
d_train = CharsDataset()
train_data = data.DataLoader(d_train, batch_size=8, shuffle=True)

model = CharsModel()
# создайте объект модели

optimizer = optim.Adam(model.parameters(), lr=0.01)
# оптимизатор Adam с шагом обучения 0.01
loss_func = nn.CrossEntropyLoss()
# функция потерь - CrossEntropyLoss

epochs = 100 # число эпох (это конечно, очень мало, в реальности нужно от 100 и более)
# переведите модель в режим обучения
model.train()
for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train)
        # вычислите прогноз модели для x_train
        loss = loss_func(predict, y_train)
        # вычислите потери для predict и y_train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # выполните один шаг обучения (градиентного спуска)

# переведите модель в режим эксплуатации
model.eval()
predict = "нейронная сеть ".lower() # начальная фраза
total = 20 # число прогнозируемых символов (дополнительно к начальной фразе)

# выполните прогноз следующих total символов
prev_chars = 10
numbers = [CharsDataset().alpha_to_int[letter] for letter in predict]
for i in range(total):
    one_hot = torch.eye(CharsDataset().count)[numbers[len(numbers) - prev_chars:]]
    predicted = model(one_hot.unsqueeze(0))
    # print(predicted)
    next_letter = predicted.argmax().item()
    numbers.append(next_letter)
# выведите полученную строку на экран
print(''.join([CharsDataset().int_to_alpha[number] for number in numbers]))

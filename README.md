# Optimize computer vision models

Практика по курсу "Оптимизация моделей компьютерного зрения"  

Задание 1.
Подготовка окружения и обучение простого классификатора на учебном датасете, например, cifar10.

Задание 2.
Сравнить скорость сходимости для разных алгоритмов оптимизации. Результат обучения в виде графиков loss/epoch отобразить на изображении и разместить в своем файле Readme. Снабдить пояснениями и выводами.

Рябинкин Валерий Николаевич	SGD, SparseAdam

Вопрос: SparseAdam предназначен для работы с разреженными градиентами в NLP (например, torch.nn.Embedding). Он не поддерживает плотные градиенты, которые создаются в моделях сверточных нейронных сетей. Возникает ошибка: RuntimeError: SparseAdam does not support dense gradients, please consider Adam instead. Чтобы устранить ошибку и продолжить обучение, я заменил SparseAdam на Adam, который работает с плотными градиентами. Можно ли было так сделать?

Я модифицировал исходный файл train. 

Добавил возможность обучать на gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Добавил модель на GPU
net = build_model().to(device)

И добавил оба алгоритма оптимизации
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

Делаяю логирование потерь
losses_per_epoch = []

Дальше в цикле обучения я добавляю явно данные в GPU
inputs, labels = inputs.to(device), labels.to(device)

Дальше считаю среднюю потерю за эпоху
avg_epoch_loss = epoch_loss / len(trainloader)
losses_per_epoch.append(avg_epoch_loss)

Отрисовываю график
plot_loss(losses_per_epoch, optimizer_name)
def plot_loss(losses, optimizer_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Loss ({optimizer_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Epoch ({optimizer_name})')
    plt.legend()
    plt.grid()
    plt.savefig(f'loss_plot_{optimizer_name.lower()}.png')
    plt.close()
    print(f"Loss plot saved as 'loss_plot_{optimizer_name.lower()}.png'")


Запускаю функцию train два раза подряд с двумя алгоритмами оптимизации
if __name__ == '__main__':
    # Train with SGD
    train(optimizer_name='SGD')

    # Train with SparseAdam
    train(optimizer_name='Adam')


В данном случае я использовал SGD и Adam, первый считается базовым методом оптимизации, его муть это обновление веса на основе градиента функции потерь. Для такой задачи он подходит хорошо, но он имеет несколько проблем: он может застревать в локальных минимумах и седловых точках, чувствителен к скорости обучения.

Adam совмещает в себе идеи AdaGrad и RMSProb. Он адаптирует learning rate для каждого параметра на основе первой и второй моментов градиентов (усреднения градиентов и их квадратов). Это позволяет ему эффективно адаптироваться как к плоским, так и к крутым областям пространства параметров.

В рамках обучения на Cifar10 сложно сказать, какой алгоритм работает лучше, они оба хорошо сходятся.
https://github.com/valeriiriabinkin/optcv24_Ryabinkin/blob/master/loss_plot_adam.png
https://github.com/valeriiriabinkin/optcv24_Ryabinkin/blob/master/loss_plot_sgd.png
Однако если сравнивать в целом по задачам глубокого обучения, то Adam используется почти всегда, если у вас одна GPU, он быстро сходится и способен стабильно работать и на большем объеме данных. Если же мы используем несколько GPU, то тут чаще может использоваться SGD, так как он обновляет параметры модели после вычисления градиентов на каждом батче. Adam, использует дополнительные состояния для каждого параметра (моменты градиента), что увеличивает накладные расходы на синхронизацию и память.

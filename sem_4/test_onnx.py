"""
https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
"""

import time
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from model import build_model
from model import classes

print(torch.__version__)
import onnx
import onnxscript
print(onnxscript.__version__)

import onnxruntime
print(onnxruntime.__version__)

onnx_model_path = 'models/cifar_model.onnx'
torch_model_path = 'models/cifar_model.pth'

batch_size = 1
torch_input = torch.randn(batch_size, 3, 32, 32)


def convert_to_onnx():
    torch_model = build_model()
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model.eval()

    # Создать папку models, если её нет
    os.makedirs('models', exist_ok=True)

    torch.onnx.export(
        torch_model,
        torch_input,
        onnx_model_path,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_model_path}")


def test_onnx_load():
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_onnx_runtime():
    torch_model = build_model()
    onnx_program = torch.onnx.export(torch_model, torch_input)
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)

    print(f"Input length: {len(onnx_input)}")
    print(f"Sample input: {onnx_input}")

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
    t0 = time.time()
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    t1 = time.time() - t0

    print('Onnx output: ', onnxruntime_outputs)
    print('Onnx time, ms: ', t1 * 1000)


def benchmark_model(model_type='torch'):
    # Загрузка CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    time_pred = []

    if model_type == 'onnx':
        ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    torch_model = build_model()
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model.eval()

    for data in testloader:
        images, labels = data

        if model_type == 'onnx':
            ort_input_name = ort_session.get_inputs()[0].name
            ort_input = {ort_input_name: to_numpy(images)}
            t0 = time.time()
            outputs = ort_session.run(None, ort_input)
            t1 = time.time() - t0
            predictions = np.argmax(outputs[0], axis=1)
        else:
            t0 = time.time()
            outputs = torch_model(images)
            t1 = time.time() - t0
            predictions = torch.argmax(outputs, dim=1)

        time_pred.append(t1 * 1000)  # время в миллисекундах

        label = labels.numpy()[0]
        if label == predictions[0]:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    accuracy = {classname: 100 * correct_pred[classname] / total_pred[classname] for classname in classes}
    mean_time = np.mean(time_pred)

    return accuracy, mean_time


if __name__ == '__main__':
    convert_to_onnx()
    print("=== PyTorch модель ===")
    torch_accuracy, torch_time = benchmark_model('torch')
    print("Точность PyTorch модели:", torch_accuracy)
    print(f"Среднее время PyTorch модели: {torch_time:.2f} ms")

    print("\n=== ONNX модель ===")
    onnx_accuracy, onnx_time = benchmark_model('onnx')
    print("Точность ONNX модели:", onnx_accuracy)
    print(f"Среднее время ONNX модели: {onnx_time:.2f} ms")


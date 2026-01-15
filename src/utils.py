import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


def dice_coefficient(preds, targets, smooth=1e-6):
    """Calcula el Dice Coefficient."""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1 - dice_score


def evaluate(model, criterion, data_loader, device):
    """
    Evalúa el modelo en los datos proporcionados y calcula la pérdida promedio.

    Args:
        model (torch.nn.Module): El modelo que se va a evaluar.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        data_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de evaluación.

    Returns:
        float: La pérdida promedio en el conjunto de datos de evaluación.

    """
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device).float()
            output = model(x)
            total_loss += criterion(output, y).item()
            total_dice += dice_coefficient(output, y).item()

    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)

    return avg_loss, avg_dice


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def print_log(epoch, train_loss, val_loss, train_dice, val_dice):
    print(
        f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Train Dice: {train_dice:.5f} | Val Dice: {val_dice:.5f}"
    )

def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    do_early_stopping=True,
    patience=5,
    epochs=10,
    log_fn=print_log,
    log_every=1,
    scheduler=None
):
    """
    Entrena el modelo utilizando el optimizador y la función de pérdida proporcionados.

    Args:
        model (torch.nn.Module): El modelo que se va a entrenar.
        optimizer (torch.optim.Optimizer): El optimizador que se utilizará para actualizar los pesos del modelo.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        train_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de entrenamiento.
        val_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de validación.
        device (str): El dispositivo donde se ejecutará el entrenamiento.
        patience (int): Número de épocas a esperar después de la última mejora en val_loss antes de detener el entrenamiento (default: 5).
        epochs (int): Número de épocas de entrenamiento (default: 10).
        log_fn (function): Función que se llamará después de cada log_every épocas con los argumentos (epoch, train_loss, val_loss) (default: None).
        log_every (int): Número de épocas entre cada llamada a log_fn (default: 1).

    Returns:
        Tuple[List[float], List[float]]: Una tupla con dos listas, la primera con el error de entrenamiento de cada época y la segunda con el error de validación de cada época.

    """
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_dice = []
    epoch_val_dice = []

    if do_early_stopping:
        early_stopping = EarlyStopping(patience=patience)  # instanciamos el early stopping

    for epoch in range(epochs):  # loop de entrenamiento
        model.train()  # ponemos el modelo en modo de entrenamiento
        train_loss = 0  # acumulador de la perdida de entrenamiento
        train_dice = 0

        for x, y in train_loader:
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device).float()  # movemos los datos al dispositivo

            optimizer.zero_grad()  # reseteamos los gradientes
            output = model(x)  # forward pass (prediccion)
            batch_loss = criterion(output, y)  # calculamos la perdida con la salida esperada
            batch_loss.backward()  # backpropagation
            optimizer.step()  # actualizamos los pesos

            train_loss += batch_loss.item()  # acumulamos la perdida
            train_dice += dice_coefficient(output, y).item()

        train_loss /= len(train_loader)  # calculamos la perdida promedio de la epoca
        train_dice /= len(train_loader)

        epoch_train_losses.append(train_loss)  # guardamos la perdida de entrenamiento
        epoch_train_dice.append(train_dice)

        val_loss, val_dice = evaluate(model, criterion, val_loader, device)  # evaluamos el modelo en el conjunto de validacion
        epoch_val_losses.append(val_loss)  # guardamos la perdida de validacion
        epoch_val_dice.append(val_dice)

        if scheduler is not None:
            scheduler.step(val_dice)

        if do_early_stopping:
            early_stopping(val_loss)  # llamamos al early stopping

        if log_fn is not None and (epoch + 1) % log_every == 0:  # si se pasa una funcion de log, loggeamos cada log_every epocas
                log_fn(epoch, train_loss, val_loss, train_dice, val_dice)  # llamamos a la funcion de log

        if do_early_stopping and early_stopping.early_stop:
            print(
                f"Detener entrenamiento en la época {epoch + 1}, la mejor pérdida fue {early_stopping.best_score:.5f}"
            )
            break

    return epoch_train_losses, epoch_val_losses, epoch_train_dice, epoch_val_dice


def plot_training(train_losses, val_losses, train_dice, val_dice):
    epochs = range(1, len(train_losses) + 1)
    tick_step = 10
    ticks = [1] + list(range(10, len(train_losses) + 1, tick_step))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de loss
    axes[0].plot(epochs, train_losses, label="Training", linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, val_losses, label="Validation", linewidth=2, marker='s', markersize=3)
    axes[0].set_title("Training and validation loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_xticks(ticks)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfico de dice
    axes[1].plot(epochs, train_dice, label="Training", linewidth=2, marker='o', markersize=3)
    axes[1].plot(epochs, val_dice, label="Validation", linewidth=2, marker='s', markersize=3)
    axes[1].set_title("Training and validation dice score", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Dice score", fontsize=12)
    axes[1].set_xticks(ticks)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    plt.show()


def model_classification_report(model, dataloader, device, nclasses):
    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcular precisión (accuracy)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Reporte de clasificación
    report = classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(nclasses)]
    )
    print("Reporte de clasificación:\n", report)


def show_tensor_image(tensor, title=None, vmin=None, vmax=None):
    """
    Muestra una imagen representada como un tensor.

    Args:
        tensor (torch.Tensor): Tensor que representa la imagen. Size puede ser (C, H, W).
        title (str, optional): Título de la imagen. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    # Check if the tensor is a grayscale image
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:  # Assume RGB
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(tensors, titles=None, figsize=(15, 5), vmin=None, vmax=None):
    """
    Muestra una lista de imágenes representadas como tensores.

    Args:
        tensors (list): Lista de tensores que representan las imágenes. El tamaño de cada tensor puede ser (C, H, W).
        titles (list, optional): Lista de títulos para las imágenes. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        # Check if the tensor is a grayscale image
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:  # Assume RGB
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()

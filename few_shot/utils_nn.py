from few_shot.utils import classification_metrics
import numpy as np
import torch
from torch import nn
torch.set_default_dtype(torch.float64)


def pd_to_torch(df, target, features):
    target = torch.tensor(df[target].values)
    feats = torch.tensor(df[features].values)
    train_tensor = torch.utils.data.TensorDataset(feats, target)

    return train_tensor


class NeuralNetwork(nn.Module):
    def __init__(self, config_hp, input_dim):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(config_hp["dropout"]),
            nn.Linear(input_dim, int(input_dim*config_hp["input_hidden_layer_dim_ratio"])),
            nn.ReLU(),
            nn.Dropout(config_hp["dropout"]),
            nn.Linear(int(input_dim*config_hp["input_hidden_layer_dim_ratio"]), 2),
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetworkTrainer():
    def __init__(self, config_hp, input_dim):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config_hp = config_hp
        self.model = NeuralNetwork(self.config_hp, input_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config_hp["lr"], momentum=self.config_hp["momentum"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.min_validation_loss = np.inf
        self.counter = 0

    def train_loop(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def val_loop(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        val_loss, correct = 0, 0
        self.model.eval()

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= num_batches
        correct /= size
        print(f"val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

        return val_loss


    def inference_loop(self, test_data, target, features, metrics):
        test_tensor = pd_to_torch(test_data, target, features)
        test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=self.config_hp["test_batch_size"], shuffle=True)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for X, y in test_loader:
                label = y
                logits = self.model(X)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                output = pred.cpu().tolist(), label.cpu().tolist(), probs.cpu().tolist()
                outputs.append(output)

        preds = []
        labels = []
        probs = []
        for pred, label, prob_1 in outputs:
            preds.extend(pred)
            labels.extend(label)
            probs.extend(prob_1)

        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric] = classification_metrics(preds, labels, probs, metric)
        return metrics_dict

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.config_hp["min_delta"]):
            self.counter += 1
            if self.counter >= self.config_hp["patience"]:
                return True
        return False
    
    def train(self, train_data, val_data, target, features):
        train_tensor = pd_to_torch(train_data, target, features)
        val_tensor = pd_to_torch(val_data, target, features)
        train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=self.config_hp["train_batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_tensor, batch_size=self.config_hp["val_batch_size"], shuffle=True)
        for e in range(self.config_hp["epochs"]):
            print(f"Epoch {e+1}\n-------------------------------")
            self.train_loop(train_loader)
            val_loss = self.val_loop(val_loader)
            if self.early_stop(val_loss):
                print("Early Stopping at epoch:", e+1)
                break
        print("Done!")
        
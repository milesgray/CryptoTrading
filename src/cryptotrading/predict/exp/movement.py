from data_provider.data_factory import data_provider
from .base import BaseExp
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from cryptotrading.predict.models import get_model
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from utils.tools import use_amp

warnings.filterwarnings("ignore")


class MovementExp(BaseExp):
    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder - decoder                
                with torch.amp.autocast(device_type=self.device, enabled=self.args.use_amp):
                    outputs, _ = self.model(batch_x, batch_x_mark)                
                    outputs = outputs[:, -1]

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                    total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        use_print = not self.args.use_tqdm
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        ebar = tqdm(range(self.args.train_epochs), disable=use_print)
        for epoch in ebar:
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            bar = tqdm(enumerate(train_loader), disable=use_print)
            for i, (batch_x, batch_y, batch_x_mark) in bar:
                metrics = {}
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                total_loss = None
                train_correct = 0
                train_total = 0
                # encoder - decoder
                with torch.amp.autocast(device_type=self.device, enabled=self.args.use_amp):
                    outputs, confidence = self.model(batch_x, batch_x_mark)
                    outputs = outputs[:, -1]

                    loss = criterion(outputs, batch_y)
                    predicted_classes = (torch.sigmoid(outputs) > 0.5).float()
                    correct = torch.eq(predicted_classes, batch_y)
                    correct_confidence = confidence[correct]
                    incorrect_confidence = confidence[~correct]
                    train_correct += correct.float().sum().item()
                    train_total += batch_y.size(0)
                    total_loss = loss
                    train_loss.append(loss.item())
                    metrics["loss"] = loss.item()
                    metrics["loss_avg"] = sum(train_loss) / len(train_loss)
                    metrics["acc"] = train_correct / train_total
                    metrics["confidence"] = confidence.mean().item()
                    metrics["correct_confidence"] = correct_confidence.mean().item()
                    metrics["incorrect_confidence"] = (
                        incorrect_confidence.mean().item()
                    )
                
                if use_print and (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    iter_count = 0
                    time_now = time.time()

                    msg = f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {metrics['loss_avg']}"
                    msg = f"{msg}\n\t\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s"
                    print(msg)
                
                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
                bar.set_postfix(metrics)

            if use_print:
                print(
                    "Epoch: {} cost time: {}".format(
                        epoch + 1, time.time() - epoch_time
                    )
                )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.test(test_data, test_loader, criterion)
            ebar.set_postfix({"vali": vali_loss, "test": test_loss})
            if use_print:
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss
                    )
                )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return {"vali_loss": vali_loss, "test_loss": test_loss}

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )
        test_correct = 0
        test_total = 0
        test_predictions = []
        test_confidences = []
        test_targets = []
        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                with torch.amp.autocast(device_type=self.device, enabled=self.args.use_amp):
                    if self.args.output_attention:
                        predictions, confidence = self.model(
                            batch_x
                        )
                    else:
                        predictions, confidence = self.model(
                            batch_x
                        )                
                predictions = predictions[:, -1]  # Get prediction for the last timestep
                confidence = confidence[:, -1]

                # Calculate accuracy
                predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
                test_correct += (predicted_classes == batch_y).sum().item()
                test_total += batch_y.size(0)

                # Store predictions and targets
                test_predictions.append(predicted_classes.numpy())
                test_confidences.append(confidence.numpy())
                test_targets.append(batch_y.numpy())

                pred = predictions
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    x = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = x.shape
                        x = test_data.inverse_transform(x.squeeze(0)).reshape(
                            shape
                        )
                    gt = np.concatenate((x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))
        test_predictions = np.concatenate(test_predictions)
        test_confidences = np.concatenate(test_confidences)
        test_targets = np.concatenate(test_targets)

        test_acc = test_correct / test_total * 100
        print(f'Test Accuracy: {test_acc:.2f}%')

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}

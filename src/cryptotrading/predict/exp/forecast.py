from cryptotrading.predict.data import data_provider
from .base import BaseExp
from cryptotrading.predict.utils.train import EarlyStopping, adjust_learning_rate, visual
from cryptotrading.predict.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ForecastExp(BaseExp):
    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _run_model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """Safely runs the forward pass of the model and unpacks outputs.
        
        Handles:
        - Linear-type models (e.g. Linear, DLinear, NLinear, WAVESTATE) which accept only 1 argument.
        - Transformer-type models (e.g. Autoformer, Transformer, Informer) which accept 4 arguments.
        - Unpacking results that can be a single tensor, a 2-tuple, or a 3-tuple.
        """
        # Determine the model inputs
        if any(m in self.args.model for m in ['Linear', 'DLinear', 'NLinear', 'WAVESTATE']):
            result = self.model(batch_x)
        else:
            result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
        # Safely unpack the result
        loss_IB = torch.tensor(0.0).to(batch_x.device)
        attn = None
        
        if isinstance(result, tuple):
            if len(result) == 3:
                outputs, loss_IB, attn = result
            elif len(result) == 2:
                outputs, second_val = result
                # Determine if second_val is attention or loss_IB
                if isinstance(second_val, torch.Tensor) and second_val.ndim == 0:
                    loss_IB = second_val
                else:
                    attn = second_val
            else:
                outputs = result[0]
        else:
            outputs = result
            
        return outputs, loss_IB, attn

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_index = batch_index.to(self.device)

                # encoder - decoder / RAFT forward
                if self.args.model == 'RAFT':
                    mode = 'valid'
                    if hasattr(vali_data, 'set_type'):
                        if vali_data.set_type == 2:
                            mode = 'test'
                        elif vali_data.set_type == 0:
                            mode = 'train'
                    outputs = self.model(batch_x, batch_index, mode=mode)
                else:
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss) if total_loss else 0.0
        self.model.train()
        return total_loss

    def test_zao(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_index = batch_index.to(self.device)

                # encoder - decoder / RAFT forward
                if self.args.model == 'RAFT':
                    stddev = 0.1
                    noisy = torch.randn_like(batch_x) * stddev
                    batch_x = batch_x + noisy
                    mode = 'valid'
                    if hasattr(vali_data, 'set_type'):
                        if vali_data.set_type == 2:
                            mode = 'test'
                        elif vali_data.set_type == 0:
                            mode = 'train'
                    outputs = self.model(batch_x, batch_index, mode=mode)
                else:
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            stddev = 0.1
                            noisy = torch.randn_like(batch_x) * stddev
                            batch_x = batch_x + noisy
                            outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        stddev = 0.1
                        noisy = torch.randn_like(batch_x) * stddev
                        batch_x = batch_x + noisy
                        outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss) if total_loss else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        use_print = not self.args.use_tqdm
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Run pre-computation phase for retrieval-augmented models like RAFT
        if hasattr(self.model, 'prepare_dataset'):
            self.model.prepare_dataset(train_data, vali_data, test_data)

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
            IB_loss = []
            PS_loss = []

            self.model.train()
            epoch_time = time.time()
            bar = tqdm(enumerate(train_loader), disable=use_print)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index) in bar:
                metrics = {}
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_index = batch_index.to(self.device)

                # encoder - decoder / RAFT forward
                if self.args.model == 'RAFT':
                    outputs = self.model(batch_x, batch_index, mode='train')
                    loss_IB = torch.tensor(0.0).to(self.device)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    total_loss = loss
                else:
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, loss_IB, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            total_loss = loss + loss_IB
                    else:
                        if self.args.ib_loss_enabled:
                            outputs, loss_IB, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            loss_IB = torch.tensor(0.0).to(self.device)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if hasattr(self.model, 'ps_loss'):
                            if self.args.ps_loss_mode == 'mean':
                                losses_ps = torch.tensor([self.model.ps_loss(outputs[:,:,i].unsqueeze(-1), batch_y[:,:,i].unsqueeze(-1)) 
                                           for i in range(outputs.shape[-1])])
                                loss_ps = losses_ps.mean()
                            elif self.args.ps_loss_mode == 'sum':
                                losses_ps = torch.tensor([self.model.ps_loss(outputs[:,:,i].unsqueeze(-1), batch_y[:,:,i].unsqueeze(-1)) 
                                           for i in range(outputs.shape[-1])])
                                loss_ps = losses_ps.sum()
                            elif self.args.ps_loss_mode == 'last':
                                loss_ps = self.model.ps_loss(outputs[:,:,-1].unsqueeze(-1),
                                                             batch_y[:,:,-1].unsqueeze(-1))    
                            else:
                                loss_ps = 0.0
                            total_loss = loss+ loss_ps
                        if self.args.ib_loss_enabled:
                            total_loss = loss + loss_IB

                train_loss.append(loss.item())
                metrics["loss"] = loss.item()
                metrics["loss_avg"] = sum(train_loss) / len(train_loss)
                
                if self.args.model != 'RAFT':
                    if self.args.ps_loss_mode in ['mean', 'sum', 'last']:
                        PS_loss.append(loss_ps.item())
                        metrics["loss_PS"] = loss_ps.item()
                        metrics["loss_PS_avg"] = sum(PS_loss) / len(PS_loss)
                    if self.args.ib_loss_enabled:
                        IB_loss.append(loss_IB.item()) 
                        metrics["loss_IB"] = loss_IB.item()
                        metrics["loss_IB_avg"] = sum(IB_loss) / len(IB_loss)
                
                metrics["total_loss"] = total_loss.item()

                if use_print and (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    iter_count = 0
                    time_now = time.time()
                    
                    msg = f"\titers: {i + 1}, epoch: {epoch + 1} | total loss: {total_loss.item()} | t loss: {metrics['loss_avg']}"
                    if self.args.model != 'RAFT':
                        if self.args.ib_loss_enabled:
                            msg = f"{msg} | ib loss: {metrics['loss_IB_avg']}"
                        if self.args.ps_loss_mode in ['mean', 'sum', 'last']:
                            msg = f"{msg} | ps loss: {metrics['loss_PS_avg']}"
                    msg = f'{msg}\n\t\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s'
                    print(msg)

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()
                bar.set_postfix(metrics)

            if use_print: print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            ebar.set_postfix({"vali": vali_loss, "test": test_loss})
            if use_print: 
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return {
            "vali_loss": vali_loss,
            "test_loss": test_loss
        }

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        # Run pre-computation phase for retrieval-augmented models like RAFT if not already done
        if hasattr(self.model, 'prepare_dataset') and not hasattr(self.model, 'retrieval_dict'):
            print("Pre-computing retrieval vectors for RAFT model during test evaluation...")
            train_data, _ = self._get_data(flag='train')
            vali_data, _ = self._get_data(flag='val')
            self.model.prepare_dataset(train_data, vali_data, test_data)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_index = batch_index.to(self.device)

                # encoder - decoder / RAFT forward
                if self.args.model == 'RAFT':
                    outputs = self.model(batch_x, batch_index, mode='test')
                else:
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _, _ = self._run_model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input_val = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_val.shape
                        input_val = test_data.inverse_transform(input_val.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_val[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_val[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return {
            "mae": mae, 
            "mse": mse, 
            "rmse": rmse, 
            "mape": mape, 
            "mspe": mspe
        }

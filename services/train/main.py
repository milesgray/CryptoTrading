import argparse
import os
import sys
import uuid
import torch
import random
import numpy as np
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from cryptotrading.predict.exp.forecast import ForecastExp
from cryptotrading.predict.exp.movement import MovementExp
from cryptotrading.predict.utils.print_args import print_args
from cryptotrading.predict.utils import dotdict
from cryptotrading.predict.models import get_model

app = FastAPI(
    title="CryptoTrading HTTP Training Service",
    description="Endpoint driven training and model serving service for models in cryptotrading.predict"
)

# Global task state
tasks = {}
tasks_lock = threading.Lock()

class TrainRequest(BaseModel):
    task_name: str = Field(default='long_term_forecast', description='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, movement]')
    model_id: str = Field(default='test', description='model id')
    model: str = Field(default='Autoformer', description='model name')
    data: str = Field(default='custom', description='dataset type')
    root_path: str = Field(default='./data/ETT/', description='root path of the data file')
    data_path: str = Field(default='ETTh1.csv', description='data file')
    features: str = Field(default='M', description='M, S, or MS')
    target: str = Field(default='OT', description='target feature')
    freq: str = Field(default='h', description='frequency for time features encoding')
    timeenc: int = Field(default=0, description='time encoding type')
    scale: bool = Field(default=True, description='whether to scale the data')
    checkpoints: str = Field(default='./checkpoints/', description='checkpoints dir')
    seq_len: int = Field(default=96)
    label_len: int = Field(default=48)
    pred_len: int = Field(default=96)
    seasonal_patterns: str = Field(default='Monthly')
    inverse: bool = Field(default=False)
    mask_rate: float = Field(default=0.25)
    anomaly_ratio: float = Field(default=0.25)
    expand: int = Field(default=2)
    d_conv: int = Field(default=4)
    top_k: int = Field(default=5)
    num_kernels: int = Field(default=6)
    enc_in: int = Field(default=7)
    dec_in: int = Field(default=7)
    c_out: int = Field(default=7)
    d_model: int = Field(default=512)
    n_heads: int = Field(default=8)
    e_layers: int = Field(default=2)
    d_layers: int = Field(default=1)
    d_ff: int = Field(default=2048)
    moving_avg: int = Field(default=25)
    factor: int = Field(default=1)
    distil: bool = Field(default=True)
    dropout: float = Field(default=0.1)
    embed: str = Field(default='timeF')
    activation: str = Field(default='gelu')
    output_attention: bool = Field(default=False)
    channel_independence: int = Field(default=1)
    decomp_method: str = Field(default='moving_avg')
    use_norm: int = Field(default=1)
    down_sampling_layers: int = Field(default=0)
    down_sampling_window: int = Field(default=1)
    down_sampling_method: Optional[str] = Field(default=None)
    seg_len: int = Field(default=48)
    num_workers: int = Field(default=10)
    itr: int = Field(default=1)
    train_epochs: int = Field(default=10)
    batch_size: int = Field(default=32)
    patience: int = Field(default=3)
    learning_rate: float = Field(default=0.0001)
    des: str = Field(default='test')
    loss: str = Field(default='MSE')
    ps_loss_mode: str = Field(default='none')
    ib_loss_enabled: bool = Field(default=True)
    lradj: str = Field(default='type1')
    use_amp: bool = Field(default=False)
    use_gpu: bool = Field(default=True)
    gpu: int = Field(default=0)
    use_multi_gpu: bool = Field(default=False)
    devices: str = Field(default='0,1,2,3')
    p_hidden_dims: List[int] = Field(default=[128, 128])
    p_hidden_layers: int = Field(default=2)
    use_tqdm: bool = Field(default=False)
    n_period: Optional[int] = Field(default=2)
    topm: Optional[int] = Field(default=5)


class PredictRequest(BaseModel):
    x: List[List[List[float]]] = Field(description="Input sequence tensor of shape [batch_size, seq_len, enc_in]")
    x_mark: Optional[List[List[List[float]]]] = Field(default=None, description="Time features for input of shape [batch_size, seq_len, num_features]")
    y_mark: Optional[List[List[List[float]]]] = Field(default=None, description="Time features for output of shape [batch_size, label_len + pred_len, num_features]")


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def run_training_task(task_id: str, request_data: TrainRequest):
    with tasks_lock:
        tasks[task_id]['status'] = 'running'
        tasks[task_id]['started_at'] = datetime.utcnow().isoformat()
    
    try:
        args_dict = request_data.dict()
        args = dotdict(args_dict)
        
        args.use_gpu = True if torch.cuda.is_available() else False
        
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
            
        if args.task_name in ('forecast', 'long_term_forecast', 'short_term_forecast'):
            if args.task_name == 'forecast':
                args.task_name = 'long_term_forecast'
            Exp = ForecastExp
        elif args.task_name == 'movement':
            Exp = MovementExp
        else:
            raise ValueError(f"Invalid task name: {args.task_name}")
            
        os.makedirs(args.checkpoints, exist_ok=True)
        
        test_metrics_list = []
        for ii in range(args.itr):
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            train_res = exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_metrics = exp.test(setting)
            test_metrics_list.append(test_metrics)
            torch.cuda.empty_cache()
            
            # Save the config.json inside the checkpoint directory
            checkpoint_dir = os.path.join(args.checkpoints, setting)
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                json.dump(args_dict, f, indent=4)
                
        with tasks_lock:
            tasks[task_id]['status'] = 'success'
            tasks[task_id]['completed_at'] = datetime.utcnow().isoformat()
            tasks[task_id]['metrics'] = make_json_serializable(test_metrics_list[-1]) if test_metrics_list else {}
            tasks[task_id]['setting'] = setting
            tasks[task_id]['checkpoint_dir'] = os.path.join(args.checkpoints, setting)
    except Exception as e:
        with tasks_lock:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['completed_at'] = datetime.utcnow().isoformat()
            tasks[task_id]['error'] = str(e)
        import traceback
        traceback.print_exc()


@app.post("/train")
def train_model_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "metrics": None,
            "error": None,
            "config": request.dict()
        }
    background_tasks.add_task(run_training_task, task_id, request)
    return {"task_id": task_id, "status": "queued"}


@app.get("/tasks")
def list_tasks():
    with tasks_lock:
        return list(tasks.values())


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return tasks[task_id]


PROJECT_ROOT = os.path.abspath(os.getcwd())
import tempfile
TEMP_DIR = os.path.abspath(tempfile.gettempdir())

def validate_safe_path(base_dir: str, relative_path: str = None) -> str:
    abs_base = os.path.abspath(base_dir)
    # Ensure base_dir is within the project root or system temp directory
    if not (abs_base.startswith(PROJECT_ROOT) or abs_base.startswith(TEMP_DIR)):
        raise HTTPException(status_code=400, detail="Access denied: Directory must be within the project root or temp directory.")
    
    if relative_path:
        abs_target = os.path.abspath(os.path.join(abs_base, relative_path))
        if not (abs_target.startswith(abs_base) or abs_target.startswith(TEMP_DIR)):
            raise HTTPException(status_code=400, detail="Access denied: Path traversal detected.")
        return abs_target
    return abs_base


@app.get("/models")
def list_models(checkpoints_dir: str = "./checkpoints/"):
    safe_checkpoints_dir = validate_safe_path(checkpoints_dir)
    if not os.path.exists(safe_checkpoints_dir):
        return []
    
    models = []
    for root, dirs, files in os.walk(safe_checkpoints_dir):
        if "checkpoint.pth" in files:
            relative_dir = os.path.relpath(root, safe_checkpoints_dir)
            model_info = {
                "model_id": relative_dir,
                "path": root,
                "has_config": "config.json" in files
            }
            if "config.json" in files:
                try:
                    with open(os.path.join(root, "config.json"), "r") as f:
                        model_info["config"] = json.load(f)
                except:
                    pass
            models.append(model_info)
    return models


@app.get("/models/{model_id:path}/download")
def download_model(model_id: str, checkpoints_dir: str = "./checkpoints/"):
    safe_checkpoints_dir = validate_safe_path(checkpoints_dir)
    checkpoint_file = validate_safe_path(safe_checkpoints_dir, os.path.join(model_id, "checkpoint.pth"))
    if not os.path.exists(checkpoint_file):
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found for {model_id}")
    return FileResponse(checkpoint_file, media_type="application/octet-stream", filename=f"{os.path.basename(model_id)}_checkpoint.pth")


@app.post("/models/{model_id:path}/predict")
def run_model_inference(model_id: str, request: PredictRequest, checkpoints_dir: str = "./checkpoints/"):
    safe_checkpoints_dir = validate_safe_path(checkpoints_dir)
    model_dir = validate_safe_path(safe_checkpoints_dir, model_id)
    if not os.path.exists(model_dir):
        # Let's search inside checkpoints
        found = False
        if os.path.exists(safe_checkpoints_dir):
            for d in os.listdir(safe_checkpoints_dir):
                if d == model_id:
                    model_dir = validate_safe_path(safe_checkpoints_dir, d)
                    found = True
                    break
        if not found:
            raise HTTPException(status_code=404, detail=f"Model directory {model_id} not found.")

    config_path = validate_safe_path(model_dir, "config.json")
    checkpoint_path = validate_safe_path(model_dir, "checkpoint.pth")
    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=400, detail="Model config.json or checkpoint.pth is missing.")
        
    try:
        with open(config_path, "r") as f:
            model_config = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read model config: {str(e)}")
        
    try:
        configs = dotdict(model_config)
        configs.use_gpu = False
        model = get_model(configs).float()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model/weights: {str(e)}")

    try:
        x_tensor = torch.FloatTensor(request.x)
        # Shape validations
        if len(x_tensor.shape) != 3:
            raise HTTPException(status_code=400, detail=f"Input must have exactly 3 dimensions [batch_size, seq_len, enc_in], got {list(x_tensor.shape)}.")
        if x_tensor.shape[1] != configs.seq_len:
            raise HTTPException(status_code=400, detail=f"Input sequence length must match model seq_len ({configs.seq_len}), got {x_tensor.shape[1]}.")
        if x_tensor.shape[2] != configs.enc_in:
            raise HTTPException(status_code=400, detail=f"Input feature dimensions must match model enc_in ({configs.enc_in}), got {x_tensor.shape[2]}.")

        batch_size = x_tensor.shape[0]
        
        with torch.no_grad():
            if any(m in configs.model for m in ['Linear', 'DLinear', 'NLinear', 'WAVESTATE']):
                result = model(x_tensor)
            else:
                if request.x_mark is None or request.y_mark is None:
                    raise HTTPException(status_code=400, detail="x_mark and y_mark are required for Transformer/Autoformer models.")
                x_mark_tensor = torch.FloatTensor(request.x_mark)
                y_mark_tensor = torch.FloatTensor(request.y_mark)
                
                dec_inp = torch.zeros((batch_size, configs.pred_len, configs.c_out)).float()
                dec_inp_full = torch.zeros((batch_size, configs.label_len + configs.pred_len, configs.c_out)).float()
                result = model(x_tensor, x_mark_tensor, dec_inp_full, y_mark_tensor)
                
            if isinstance(result, tuple):
                outputs = result[0]
            else:
                outputs = result
                
            predictions = outputs.cpu().tolist()
            return {"predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


def run_cli():
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding type')
    parser.add_argument('--scale', type=bool, default=True, help='whether to scale data')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--ps_loss_mode', type=str, default='none', help='how to calculate patchwise structural loss. options: ["mean", "sum", "last", "none"]')
    parser.add_argument('--ib_loss_enabled', type=bool, default=True, help='info bottleneck loss applied')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    parser.add_argument('--use_tqdm', type=bool, default=False, help='use progress bar')
    
    # RAFT specific
    parser.add_argument('--n_period', type=int, default=2)
    parser.add_argument('--topm', type=int, default=5)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name in ('forecast', 'long_term_forecast', 'short_term_forecast'):
        if args.task_name == 'forecast':
            args.task_name = 'long_term_forecast'
        Exp = ForecastExp
    elif args.task_name == 'movement':
        Exp = MovementExp
    else:
        raise ValueError(f"Invalid task name: {args.task_name}")

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_metrics = exp.test(setting)
            torch.cuda.empty_cache()
            
            # Save configs
            checkpoint_dir = os.path.join(args.checkpoints, setting)
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                json.dump(vars(args), f, indent=4)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Determine whether to run CLI or FastAPI server.
    # If standard parameters are provided (typically --task_name, --model_id, etc. or --cli is present)
    # we run the CLI, otherwise we start the server.
    has_cli_args = any(arg.startswith('--') for arg in sys.argv[1:])
    if has_cli_args or '--cli' in sys.argv:
        if '--cli' in sys.argv:
            sys.argv.remove('--cli')
        run_cli()
    else:
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)

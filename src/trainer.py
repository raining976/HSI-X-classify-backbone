from torch.backends import cudnn

from .data import *

import math
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from . import config
from .model_registry import get_model_adapter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class EarlyStopping:
    def __init__(self, enabled=False, patience=10, min_delta=0.005):
        self.enabled = enabled
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0

    def should_stop(self, metric):
        if not self.enabled:
            return False
        if self.best is None or metric - self.best >= self.min_delta:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def getLog(log_path, str):
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')


def _build_optimizer(net, lr):
    optimizer_name = (config.get_value('optimizer_name') or 'adam').lower()
    weight_decay = config.get_value('weight_decay') or 0.0
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(net.parameters(), lr=lr)


def _build_lr_scheduler(optimizer, epochs, warmup_epochs, min_lr):
    warmup_epochs = max(0, min(warmup_epochs, max(epochs - 1, 0)))

    def lr_lambda(epoch):
        if epochs <= 1:
            return 1.0
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        cosine_total = max(1, epochs - warmup_epochs)
        cosine_epoch = max(0, epoch - warmup_epochs)
        min_factor = min_lr if min_lr > 0 else 0.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _format_parameter_count(net):
    total_params = sum(param.numel() for param in net.parameters())
    trainable_params = sum(param.numel() for param in net.parameters() if param.requires_grad)
    return (
        'model parameters: '
        f'total={total_params:,} ({total_params / 1_000_000:.3f}M), '
        f'trainable={trainable_params:,} ({trainable_params / 1_000_000:.3f}M)'
    )


def _format_large_count(value):
    if value >= 1_000_000_000:
        return f'{value / 1_000_000_000:.3f}G'
    if value >= 1_000_000:
        return f'{value / 1_000_000:.3f}M'
    if value >= 1_000:
        return f'{value / 1_000:.3f}K'
    return str(value)


def _is_mamba_module(module):
    module_path = module.__class__.__module__
    return module.__class__.__name__ == 'Mamba' and module_path.startswith('mamba_ssm.')


def _estimate_mamba_macs(module, inputs):
    if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 3:
        return 0

    tokens = inputs[0]
    token_count = tokens.numel() // tokens.shape[-1]
    d_model = tokens.shape[-1]
    d_inner = module.d_inner
    d_state = module.d_state
    d_conv = module.d_conv
    dt_rank = module.dt_rank

    in_projection = token_count * d_model * (2 * d_inner)
    depthwise_convolution = token_count * d_inner * d_conv
    state_projection = token_count * d_inner * (dt_rank + 2 * d_state)
    time_projection = token_count * dt_rank * d_inner
    selective_scan = token_count * 4 * d_inner * d_state
    output_gate = token_count * d_inner
    out_projection = token_count * d_inner * d_model
    return (
        in_projection
        + depthwise_convolution
        + state_projection
        + time_projection
        + selective_scan
        + output_gate
        + out_projection
    )


def _estimate_model_compute(model_adapter, model_bundle, net, train_loader, device):
    conv_linear_macs = 0
    mamba_macs = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal conv_linear_macs
        if not torch.is_tensor(output):
            return
        kernel_ops = module.weight.shape[2:].numel() * (module.in_channels // module.groups)
        conv_linear_macs += output.numel() * kernel_ops

    def linear_hook(module, inputs, output):
        nonlocal conv_linear_macs
        if torch.is_tensor(output):
            conv_linear_macs += output.numel() * module.in_features

    def mamba_hook(module, inputs, _output):
        nonlocal mamba_macs
        mamba_macs += _estimate_mamba_macs(module, inputs)

    mamba_modules = [module for module in net.modules() if _is_mamba_module(module)]
    mamba_child_ids = {
        id(child)
        for mamba_module in mamba_modules
        for child in mamba_module.modules()
        if child is not mamba_module
    }

    for module in net.modules():
        if _is_mamba_module(module):
            handles.append(module.register_forward_hook(mamba_hook))
        elif id(module) in mamba_child_ids:
            continue
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = net.training
    error = None
    try:
        hsi_pca, hsi, sar, tr_labels = next(iter(train_loader))
        batch = {
            "hsi_pca": hsi_pca[:1].to(device),
            "hsi": hsi[:1].to(device),
            "aux": sar[:1].to(device),
            "label": tr_labels[:1].to(device),
        }
        net.eval()
        with torch.no_grad():
            model_adapter.forward_train(model_bundle, batch)
    except Exception as exc:
        error = exc
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            net.train()

    if error is not None:
        return f'model compute: unavailable ({error})'

    macs = conv_linear_macs + mamba_macs
    flops = macs * 2
    return (
        'model compute: '
        f'MACs={_format_large_count(macs)}, '
        f'FLOPs={_format_large_count(flops)} '
        f'(per sample, approximate; conv/linear={_format_large_count(conv_linear_macs)}, '
        f'Mamba={_format_large_count(mamba_macs)})'
    )


def _atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def train(epochs, lr, model, cuda, train_loader, test_loader, out_features, model_savepath, log_path, hsi_pca_wight, datasetType):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    cudnn.benchmark = True

    hsi_pca_wight_tensor = torch.from_numpy(hsi_pca_wight).to(device)

    model_adapter = get_model_adapter(model)
    model_bundle = model_adapter.build_model(config, datasetType, device)
    net = model_bundle["net"]
    net.to(device)
    param_count_log = _format_parameter_count(net)
    compute_log = _estimate_model_compute(model_adapter, model_bundle, net, train_loader, device)
    print(param_count_log)
    print(compute_log)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get_value('label_smoothing') or 0.0)
    optimizer = _build_optimizer(net, lr)
    warmup_epochs = config.get_value('warmup_epochs') or 0
    min_lr = config.get_value('min_lr') or 0.0
    use_scheduler = warmup_epochs > 0 or min_lr > 0
    scheduler = None
    if use_scheduler:
        scheduler = _build_lr_scheduler(
            optimizer,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            min_lr=max(min_lr / max(lr, 1e-12), 0.0),
        )
    max_acc = 0
    early_stopping = EarlyStopping(
        enabled=bool(config.get_value('enable_early_stopping')),
        patience=config.get_value('early_stopping_patience') or 10,
        min_delta=config.get_value('early_stopping_min_delta') or 0.005,
    )
    sum_time = 0

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, config.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)
    getLog(log_path, param_count_log)
    getLog(log_path, compute_log)

    config.set_value('actual_epoch_nums', 0)
    for epoch in range(epochs):
        since = time.time()
        net.train()

        try:
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        except Exception:
            iterator = train_loader
        for i, (hsi_pca, hsi, sar, tr_labels) in enumerate(iterator):
            batch = {
                "hsi_pca": hsi_pca.to(device),
                "hsi": hsi.to(device),
                "aux": sar.to(device),
                "label": tr_labels.to(device),
            }

            optimizer.zero_grad()
            outputs = model_adapter.forward_train(model_bundle, batch)
            loss = criterion(outputs, batch["label"])

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if epoch % 1 == 0:
            net.eval()
            count = 0

            for hsi_pca, hsi, sar, gtlabels in test_loader:
                batch = {
                    "hsi_pca": hsi_pca.to(device),
                    "hsi": hsi.to(device),
                    "aux": sar.to(device),
                    "label": gtlabels,
                }

                with torch.no_grad():
                    outputs = model_adapter.forward_eval(model_bundle, batch)

                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                if count == 0:
                    y_pred_test = outputs
                    gty = gtlabels
                    count = 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, outputs))
                    gty = np.concatenate((gty, gtlabels))

            acc1 = accuracy_score(gty, y_pred_test)
            config.set_value('actual_epoch_nums', epoch + 1)

            if acc1 > max_acc:
                save_obj = model_bundle if set(model_bundle.keys()) != {"net"} else net
                _atomic_torch_save(save_obj, model_savepath)
                max_acc = acc1

            time_elapsed = time.time() - since
            sum_time += time_elapsed
            rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)
            currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            log = currentTime + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] [lr: %.7f] [current loss: %.4f] acc: %.4f' %(epoch + 1, time_elapsed, (rest_time // 60) // 60, (rest_time // 60) % 60, rest_time % 60, optimizer.param_groups[0]['lr'], loss.item(), acc1)
            print(log)
            getLog(log_path, log)
            if early_stopping.should_stop(acc1):
                stop_log = 'Early stopping triggered at epoch {} with acc: {:.4f}'.format(epoch + 1, acc1)
                print(stop_log)
                getLog(log_path, stop_log)
                break

    print('max_acc: %.4f' %(max_acc))
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')


def myTrain(datasetType, model):
    """训练函数，从配置系统获取参数"""
    channels = config.get_value('channels')
    windowSize = config.get_value('windowSize')
    out_features = config.get_value('out_features')
    cuda = config.get_value('cuda')
    lr = config.get_value('lr')
    epoch_nums = config.get_value('epoch_nums')
    batch_size = config.get_value('batch_size')
    num_workers = config.get_value('num_workers')
    random_seed = config.get_value('random_seed')
    model_savepath = config.get_value('model_savepath')
    log_path = config.get_value('log_path')

    print(f"训练参数: model={model}, epochs={epoch_nums}, lr={lr}, cuda={cuda}")
    set_random_seed(random_seed)
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)

    train(epoch_nums, lr, model, cuda, train_loader, test_loader, out_features[datasetType], model_savepath[datasetType], log_path[datasetType], hsi_pca_wight, datasetType)

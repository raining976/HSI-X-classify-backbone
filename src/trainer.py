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
    print(param_count_log)

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
                os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
                torch.save(net, model_savepath)
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

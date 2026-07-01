import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. 网络定义：CNN 特征提取器 + MLP 分类器
# ============================================================

class Net(nn.Module):
    """
    MNIST CNN 分类网络。
    输入: x.shape = [batch, 1, 28, 28]
    输出: logits.shape = [batch, 10]
    """

    def __init__(self):
        super().__init__()

        # CNN 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25)          # 随机失活一部分卷积特征，缓解过拟合
        )

        # MLP 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 12 * 12, out_features=128),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


# ============================================================
# 2. 训练一个 epoch
# ============================================================

def train_one_epoch(args, model, device, train_loader, criterion, optimizer, epoch):
    """
    在训练集上训练一个 epoch。
    会执行: 前向传播、loss 计算、反向传播、参数更新
    返回:
        avg_loss: 当前 epoch 的平均训练损失
        accuracy: 当前 epoch 的训练准确率
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 1. 清空上一轮梯度
        optimizer.zero_grad()

        # 2. 前向传播，输出 logits
        logits = model(data)

        # 3. 计算分类损失
        loss = criterion(logits, target)

        # 4. 反向传播，计算梯度
        loss.backward()

        # 5. 参数更新
        optimizer.step()

        # 6. 统计 loss 和 accuracy
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_samples += batch_size

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * batch_size}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)] "
                f"Loss: {loss.item():.6f}"
            )

            # dry-run 用于快速检查代码能不能跑通
            if args.dry_run:
                break

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    return avg_loss, accuracy


# ============================================================
# 3. 评估函数：可用于验证集或测试集
# ============================================================

def evaluate(model, device, data_loader, criterion, split_name="Validation"):
    """
    在验证集或测试集上评估模型。
    不会执行: 反向传播、参数更新
    返回:
        avg_loss: 平均损失
        accuracy: 准确率
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            logits = model(data)
            loss = criterion(logits, target)

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size

            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    print(
        f"{split_name} set: "
        f"Average loss: {avg_loss:.4f}, "
        f"Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)"
    )

    return avg_loss, accuracy


# ============================================================
# 4. 构建 DataLoader：训练集 / 验证集 / 测试集
# ============================================================

def build_dataloaders(args, device):
    """
    加载 MNIST，并将原始训练集进一步拆分为:
        train_set: 真正用于训练
        val_set:   每个 epoch 后用于验证
        test_set:  训练完成后最终测试

    MNIST 原始划分:
        train=True:  60000 张
        train=False: 10000 张
    这里默认从 60000 张训练图像中拆出 10000 张作为验证集，剩下 50000 张作为训练集
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    val_size = args.val_size
    train_size = len(full_train_dataset) - val_size

    if train_size <= 0:
        raise ValueError(f"val_size={val_size} 太大，训练集总大小只有 {len(full_train_dataset)}")

    # 使用固定 seed 拆分，保证每次 train / val 划分一致
    generator = torch.Generator().manual_seed(args.seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # DataLoader 通用加速参数
    # 对 MNIST 这种小数据集，num_workers=0 也完全可以。
    loader_kwargs = {"num_workers": args.num_workers}

    # GPU 训练时，pin_memory=True 通常可以加快 CPU -> GPU 的数据拷贝
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True

    # persistent_workers 只有 num_workers > 0 时才能使用
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ============================================================
# 5. 设备选择
# ============================================================

def get_device(args):
    """
    选择训练设备。
    优先级: CUDA GPU > CPU
    """

    if not args.no_accel and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    return device


# ============================================================
# 6. 主函数
# ============================================================

def main():
    # ----------------------------
    # A. 命令行参数
    # ----------------------------
    parser = argparse.ArgumentParser(description="PyTorch MNIST CNN Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
        help="训练阶段的 batch size，默认 64")

    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
        help="验证 / 测试阶段的 batch size，默认 1000")

    parser.add_argument("--epochs", type=int, default=10, metavar="N",
        help="训练 epoch 数，默认 4")

    parser.add_argument("--lr", type=float, default=1.0, metavar="LR",
        help="学习率，默认 1.0。这里配合 Adadelta 使用")

    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
        help="StepLR 学习率衰减系数，默认 0.7")

    parser.add_argument("--val-size", type=int, default=10000, metavar="N",
        help="从原始训练集中划分多少样本作为验证集，默认 10000")

    parser.add_argument("--data-dir", type=str, default="../data",
        help="MNIST 数据保存路径，默认 ../data")

    parser.add_argument("--num-workers", type=int, default=1,
        help="DataLoader 的子进程数量，默认 0。Windows 初学阶段建议先用 0")

    parser.add_argument("--no-accel", action="store_true",
        help="禁用 GPU / 加速器，强制使用 CPU")

    parser.add_argument("--dry-run", action="store_true",
        help="快速跑一个 batch，用于检查代码是否能跑通")

    parser.add_argument("--seed", type=int, default=1, metavar="S",
        help="随机种子，默认 1")

    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
        help="每隔多少个 batch 打印一次训练日志，默认 10")

    parser.add_argument("--save-model", action="store_true",
        help="是否保存验证集表现最好的模型")

    args = parser.parse_args()

    # B. 固定随机种子
    torch.manual_seed(args.seed)

    # C. 设备选择
    device = get_device(args)
    print(f"Using device: {device}")

    # D. 数据准备：train / val / test
    train_loader, val_loader, test_loader = build_dataloaders(args, device)
    print(
        f"Dataset split: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    # ----------------------------
    # E. 模型、损失函数、优化器、学习率调度器
    # ----------------------------
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # ----------------------------
    # F. 训练 + 验证
    # ----------------------------
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    print("\n=== Start Training ===")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            args=args,
            model=model,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch
        )

        val_loss, val_acc = evaluate(
            model=model,
            device=device,
            data_loader=val_loader,
            criterion=criterion,
            split_name="Validation"
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] Summary: "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        # 根据验证集 loss 保存当前最好的模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

            if args.save_model:
                torch.save(best_model_state, "mnist_cnn_best.pt")
                print("Best model updated and saved to mnist_cnn_best.pt")

        # 每个 epoch 结束后更新学习率
        scheduler.step()

        print("-" * 80)

    # ----------------------------
    # G. 训练完成后，在测试集上做最终评估
    # ----------------------------
    print("\n=== Final Evaluation on Test Set ===")

    # 使用验证集表现最好的模型，而不是最后一个 epoch 的模型
    model.load_state_dict(best_model_state)

    test_loss, test_acc = evaluate(
        model=model,
        device=device,
        data_loader=test_loader,
        criterion=criterion,
        split_name="Test"
    )

    print(
        f"Final Test Result: "
        f"Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_acc:.2f}%"
    )

    # 如果希望保存最终用于部署/测试的最佳模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn_final.pt")
        print("Final best model saved to mnist_cnn_final.pt")


if __name__ == "__main__":
    main()
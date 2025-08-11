import re

import matplotlib.pyplot as plt

# 原始日志文本（节选）
log_text = """
Train 1 epoch, Loss: 1.7276, Train_Acc: 0.2942, Test_Acc:0.3683, LR:0.000100
Train 2 epoch, Loss: 1.5015, Train_Acc: 0.4002, Test_Acc:0.4419, LR:0.000100
Train 3 epoch, Loss: 1.4057, Train_Acc: 0.4359, Test_Acc:0.4522, LR:0.000100
Train 4 epoch, Loss: 1.3408, Train_Acc: 0.4646, Test_Acc:0.4843, LR:0.000100
Train 5 epoch, Loss: 1.2771, Train_Acc: 0.4982, Test_Acc:0.5091, LR:0.000100
Train 6 epoch, Loss: 1.1995, Train_Acc: 0.5330, Test_Acc:0.5241, LR:0.000100
Train 7 epoch, Loss: 1.1380, Train_Acc: 0.5605, Test_Acc:0.5341, LR:0.000100
Train 8 epoch, Loss: 1.0766, Train_Acc: 0.5845, Test_Acc:0.5606, LR:0.000100
Train 9 epoch, Loss: 1.0141, Train_Acc: 0.6133, Test_Acc:0.5600, LR:0.000100
Train 10 epoch, Loss: 0.9431, Train_Acc: 0.6437, Test_Acc:0.5826, LR:0.000100
Train 11 epoch, Loss: 0.7481, Train_Acc: 0.7247, Test_Acc:0.6174, LR:0.000010
Train 12 epoch, Loss: 0.6740, Train_Acc: 0.7555, Test_Acc:0.6169, LR:0.000010
Train 13 epoch, Loss: 0.6248, Train_Acc: 0.7759, Test_Acc:0.6082, LR:0.000010
Train 14 epoch, Loss: 0.5873, Train_Acc: 0.7902, Test_Acc:0.6094, LR:0.000010
Train 15 epoch, Loss: 0.5465, Train_Acc: 0.8069, Test_Acc:0.6119, LR:0.000010
Train 16 epoch, Loss: 0.5047, Train_Acc: 0.8229, Test_Acc:0.6160, LR:0.000010
Train 17 epoch, Loss: 0.4718, Train_Acc: 0.8372, Test_Acc:0.6147, LR:0.000010
Train 18 epoch, Loss: 0.4320, Train_Acc: 0.8500, Test_Acc:0.6169, LR:0.000010
Train 19 epoch, Loss: 0.3982, Train_Acc: 0.8654, Test_Acc:0.6116, LR:0.000010
Train 20 epoch, Loss: 0.3702, Train_Acc: 0.8738, Test_Acc:0.6121, LR:0.000010
Train 21 epoch, Loss: 0.3216, Train_Acc: 0.8958, Test_Acc:0.6149, LR:0.000001
Train 22 epoch, Loss: 0.3159, Train_Acc: 0.8958, Test_Acc:0.6135, LR:0.000001
Train 23 epoch, Loss: 0.3080, Train_Acc: 0.8990, Test_Acc:0.6124, LR:0.000001
Train 24 epoch, Loss: 0.3025, Train_Acc: 0.9004, Test_Acc:0.6116, LR:0.000001
Train 25 epoch, Loss: 0.2967, Train_Acc: 0.9041, Test_Acc:0.6158, LR:0.000001
Train 26 epoch, Loss: 0.2863, Train_Acc: 0.9076, Test_Acc:0.6113, LR:0.000001
Train 27 epoch, Loss: 0.2867, Train_Acc: 0.9064, Test_Acc:0.6138, LR:0.000001
Train 28 epoch, Loss: 0.2875, Train_Acc: 0.9092, Test_Acc:0.6135, LR:0.000001
Train 29 epoch, Loss: 0.2819, Train_Acc: 0.9091, Test_Acc:0.6135, LR:0.000001
Train 30 epoch, Loss: 0.2870, Train_Acc: 0.9080, Test_Acc:0.6135, LR:0.000001
Train 31 epoch, Loss: 0.2753, Train_Acc: 0.9111, Test_Acc:0.6124, LR:0.000001
Train 32 epoch, Loss: 0.2758, Train_Acc: 0.9111, Test_Acc:0.6116, LR:0.000001
Train 33 epoch, Loss: 0.2664, Train_Acc: 0.9143, Test_Acc:0.6124, LR:0.000001
Train 34 epoch, Loss: 0.2637, Train_Acc: 0.9146, Test_Acc:0.6138, LR:0.000001
Train 35 epoch, Loss: 0.2648, Train_Acc: 0.9152, Test_Acc:0.6124, LR:0.000001
Train 36 epoch, Loss: 0.2610, Train_Acc: 0.9163, Test_Acc:0.6133, LR:0.000001
Train 37 epoch, Loss: 0.2591, Train_Acc: 0.9171, Test_Acc:0.6130, LR:0.000001
Train 38 epoch, Loss: 0.2507, Train_Acc: 0.9210, Test_Acc:0.6163, LR:0.000001
Train 39 epoch, Loss: 0.2487, Train_Acc: 0.9194, Test_Acc:0.6152, LR:0.000001
Train 40 epoch, Loss: 0.2472, Train_Acc: 0.9233, Test_Acc:0.6177, LR:0.000001
Train 41 epoch, Loss: 0.2475, Train_Acc: 0.9205, Test_Acc:0.6149, LR:0.000001
Train 42 epoch, Loss: 0.2457, Train_Acc: 0.9224, Test_Acc:0.6169, LR:0.000001
Train 43 epoch, Loss: 0.2462, Train_Acc: 0.9218, Test_Acc:0.6135, LR:0.000001
Train 44 epoch, Loss: 0.2408, Train_Acc: 0.9240, Test_Acc:0.6138, LR:0.000001
Train 45 epoch, Loss: 0.2378, Train_Acc: 0.9241, Test_Acc:0.6135, LR:0.000001
Train 46 epoch, Loss: 0.2345, Train_Acc: 0.9264, Test_Acc:0.6152, LR:0.000001
Train 47 epoch, Loss: 0.2305, Train_Acc: 0.9277, Test_Acc:0.6110, LR:0.000001
Train 48 epoch, Loss: 0.2240, Train_Acc: 0.9300, Test_Acc:0.6149, LR:0.000001
Train 49 epoch, Loss: 0.2265, Train_Acc: 0.9295, Test_Acc:0.6147, LR:0.000001
Train 50 epoch, Loss: 0.2220, Train_Acc: 0.9290, Test_Acc:0.6144, LR:0.000001
"""

# 提取 loss, train_acc, test_acc
losses = [float(m) for m in re.findall(r"Loss:\s([\d.]+)", log_text)]
train_accs = [float(m) for m in re.findall(r"Train_Acc:\s([\d.]+)", log_text)]
test_accs = [float(m) for m in re.findall(r"Test_Acc:([\d.]+)", log_text)]
epochs = list(range(1, len(losses) + 1))

# 单独绘制并保存 Loss 曲线图
plt.figure(figsize=(6, 5))
plt.plot(epochs, losses, color="tab:red", linewidth=2, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./loss_curve.png")

# 单独绘制并保存 Accuracy 曲线图
plt.figure(figsize=(6, 5))
plt.plot(epochs, train_accs, label="Train Accuracy", color="tab:blue", linewidth=2)
plt.plot(epochs, test_accs, label="Test Accuracy", color="tab:green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Testing Accuracy over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./accuracy_curve.png")

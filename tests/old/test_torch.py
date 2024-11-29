import torch
import torch.nn as nn
import torch.optim as optim

# 检查 CUDA 是否可用，并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型并将其转移到设备
model = SimpleModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建一些随机输入数据和标签，并将其转移到 GPU 上
inputs = torch.randn(5, 10).to(device)
labels = torch.randn(5, 1).to(device)

# 模拟一个训练步骤
for epoch in range(5):
    optimizer.zero_grad()  # 清除梯度
    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("训练完成")

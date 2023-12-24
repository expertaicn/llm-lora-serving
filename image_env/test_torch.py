import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device {device}')

x = torch.randn(10000, 10000)
y = torch.randn(10000, 2000)

x = x.to(device)
y = y.to(device)

start_time = time.time()
result = torch.matmul(x,y)
print(f'GPU time: ', time.time() - start_time)

x = x.cpu()
y = y.cpu()
start_time = time.time()
result = torch.matmul(x,y)
print('CPU time: ', time.time() - start_time)

print(torch.version.cuda)

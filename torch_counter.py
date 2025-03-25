import torch, time
counter = torch.tensor(0, device='cuda')
while True:
    print(counter)
    counter.add_(1)
    time.sleep(1)

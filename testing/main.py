import torch

weight = torch.zeros(1,requires_grad = True)

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

yhat = w * x

loss = (yhat - y)**2

print(loss)


#backward pass:
loss.backward()
print(w.grad)


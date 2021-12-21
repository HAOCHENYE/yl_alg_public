import torch

def test_clone():
    x = torch.tensor(10.0).requires_grad_()
    w = torch.tensor(5.0).requires_grad_()
    y = w*x.clone()
    y.backward()
    print(w.grad)


def test_view():
    x = torch.tensor(10.0).requires_grad_()
    w = torch.tensor(5.0).requires_grad_()
    y = w*x.view(1, 1)
    y.sum().backward()
    print(f"reshape grad is {w.grad}")

def test_reshape():
    x = torch.randn(1, 4, 4, 4).permute(0, 2, 3, 1).reshape(1, 32, 2).requires_grad_()
    w = torch.tensor(5.0).requires_grad_()
    y = w*x
    y.sum().backward()
    print(f"reshape grad is {w.grad}")

if __name__ == "__main__":
    test_clone()
    test_view()
    test_reshape()

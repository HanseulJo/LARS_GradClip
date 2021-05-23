import torch
from torch import optim
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    The LR decay feature is separated from the original code:
        https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, eta=0.01):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eta=eta)
        
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p.data)
                
                if weight_decay != 0:
                    grad_norm.add_(weight_norm, alpha=weight_decay)
                    d_p.add_(p.data, alpha=weight_decay)
                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / (grad_norm + 1e-8)
                # Update the momentum term
                actual_lr = local_lr * lr

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=actual_lr)
                p.data.add_(-buf)

        return loss

class GradClip(Optimizer):
    r"""Implements Gradient Clipping for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        threshold (float, optional): gradient norm threshold
    """
    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, threshold=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if threshold < 0.0:
            raise ValueError(f"Invalid grad norm threshold: {threshold}")
        
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        threshold=threshold)
        super(GradClip, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            threshold = group['threshold']
            lr = group['lr']
            
            device = group['params'][0].grad.device
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) \
                                                for p in group['params'] if p.grad is not None]))
            if grad_norm.isnan() or grad_norm.isinf():
                raise RuntimeError(f'The total norm for gradients from is non-finite, so it cannot be clipped.')

            if grad_norm > threshold:
                actual_lr = lr * threshold / (grad_norm + 1e-8)
            else:
                actual_lr = lr
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=actual_lr)
                p.data.add_(-buf)

        return loss

class LGC(Optimizer):
    r"""Implements Layerwise Gradient Clipping for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        threshold (float, optional): gradient norm threshold
    """
    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, threshold=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if threshold < 0.0:
            raise ValueError(f"Invalid grad norm threshold: {threshold}")
        
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        threshold=threshold)
        super(LGC, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            threshold = group['threshold']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                grad_norm = torch.norm(d_p.data)
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                if grad_norm > threshold:
                    actual_lr = lr * threshold / (grad_norm + 1e-8)
                else:
                    actual_lr = lr

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=actual_lr)
                p.data.add_(-buf)

        return loss

class LaRSPaG(Optimizer):
    r"""Implements Layerwise Rate Scaling by Parameter and Gradient  for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        threshold (float, optional): gradient norm threshold
    """
    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, eta=0.01, threshold=1.):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if threshold < 0.0:
            raise ValueError(f"Invalid grad norm threshold: {threshold}")
        if eta < 0.0:
            raise ValueError(f"Invalid LARS coefficient value: {eta}")
        
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        threshold=threshold, eta=eta)
        super(LaRSPaG, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            threshold = group['threshold']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p.data)
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                if grad_norm > threshold:
                    actual_lr = lr * (threshold + eta * weight_norm) / (grad_norm + 1e-8)
                else:
                    actual_lr = lr * eta * weight_norm / (grad_norm + 1e-8)

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=actual_lr)
                p.data.add_(-buf)

        return loss

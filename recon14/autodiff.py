"""Hand-written reverse-mode autodiff engine.

Reconstruction category: RECOVERED (specification). The historical
``phase1/autodiff.py`` is unrecoverable; this reimplementation follows the stated
contract: ``Value`` with operator overloading, ``Adam``, and ``bce_loss``. Gradients
are verified against finite differences in the test suite.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple


class Value:
    """A scalar holding a value and its local gradient bookkeeping."""

    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data: float, _prev: Sequence["Value"] = (),
                 requires_grad: bool = True) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple["Value", ...] = tuple(_prev)
        # False on frozen/embedding-constant leaves. The reverse-mode sweep
        # reaches a leaf through its *parent's* closure, not through the leaf's
        # own ``_prev``/``_backward``, so detaching a leaf is not enough to stop
        # gradient flow: every accumulation site must consult this flag.
        self.requires_grad = bool(requires_grad)

    # ---- arithmetic -------------------------------------------------
    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Value":
        return self + other

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other) -> "Value":
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) - self

    def __truediv__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1.0

    def __rtruediv__(self, other) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) / self

    def __pow__(self, p: float) -> "Value":
        assert isinstance(p, (int, float)), "only constant powers supported"
        out = Value(self.data**p, (self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += p * (self.data ** (p - 1)) * out.grad

        out._backward = _backward
        return out

    # ---- primitives --------------------------------------------------
    def exp(self) -> "Value":
        d = math.exp(self.data)
        out = Value(d, (self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += d * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0.0:
            raise ValueError("log domain error")
        out = Value(math.log(self.data), (self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        x = self.data
        if x >= 0.0:
            z = math.exp(-x)
            d = 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            d = z / (1.0 + z)
        out = Value(d, (self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += d * (1.0 - d) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        d = math.tanh(self.data)
        out = Value(d, (self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - d * d) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        d = max(0.0, self.data)
        out = Value(d, (self,))

        def _backward() -> None:
            if self.data > 0.0 and self.requires_grad:
                self.grad += out.grad

        out._backward = _backward
        return out

    # ---- graph ops ---------------------------------------------------
    def zero_grad(self) -> None:
        self.grad = 0.0

    def backward(self) -> None:
        """Reverse-mode sweep. Builds the topological order lazily."""
        topo: List[Value] = []
        visited: Set[int] = set()

        def build(v: Value) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def parameters(self) -> List["Value"]:
        """All reachable leaf Values, deduplicated by identity."""
        seen: Set[int] = set()
        out: List[Value] = []

        def walk(v: Value) -> None:
            if id(v) in seen:
                return
            seen.add(id(v))
            if not v._prev:
                out.append(v)
            else:
                for child in v._prev:
                    walk(child)

        walk(self)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


class Adam:
    """Adam (Kingma & Ba) over ``Value`` leaves with bias correction."""

    def __init__(self, params: Iterable[Value], lr: float = 0.1,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        self.params: List[Value] = list(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m: List[float] = [0.0] * len(self.params)
        self.v: List[float] = [0.0] * len(self.params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self, grad_norm_clip: Optional[float] = 1.0) -> float:
        grads = [p.grad for p in self.params]
        if grad_norm_clip is not None:
            # Global L2 norm rescaling (never magnifies).
            n = math.sqrt(sum(g * g for g in grads))
            if n > grad_norm_clip and n > 0.0:
                scale = grad_norm_clip / n
                grads = [g * scale for g in grads]
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        bc1 = 1.0 - b1**self.t
        bc2 = 1.0 - b2**self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + eps)
        return float(math.sqrt(sum(g * g for g in grads)))

    def set_params(self, params: Iterable[Value]) -> None:
        """Rebind the optimised leaves, resetting Adam state."""
        self.params = list(params)
        self.t = 0
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)


def bce_loss(pred: Value, target: float) -> Value:
    """Binary cross-entropy with clipping for log-domain safety."""
    eps = 1e-7
    lo, hi = eps, 1.0 - eps
    if pred.data <= lo:
        clamped: Value = Value(lo)
    elif pred.data >= hi:
        clamped = Value(hi)
    else:
        clamped = pred
    one = Value(1.0)
    return -(Value(target) * clamped.log() + (one - Value(target)) * (one - clamped).log())


def numerical_gradient_check(
    f: Callable[[List[Value]], Value],
    params: List[Value],
    eps: float = 1e-5,
    tol: float = 1e-3,
) -> bool:
    """Finite-difference verification of reverse-mode gradients."""
    n = len(params)
    num = [0.0] * n
    for i in range(n):
        old = params[i].data
        params[i].data = old + eps
        f_plus = f(params).data
        params[i].data = old - eps
        f_minus = f(params).data
        params[i].data = old
        num[i] = (f_plus - f_minus) / (2.0 * eps)
    for p in params:
        p.zero_grad()
    f(params).backward()
    for i, p in enumerate(params):
        if abs(p.grad - num[i]) > tol * max(1.0, abs(num[i])):
            return False
    return True


__all__ = ["Value", "Adam", "bce_loss", "numerical_gradient_check"]

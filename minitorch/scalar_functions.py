from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of two inputs."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the output with respect to the inputs."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the logarithm of the input."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the logarithm function with respect to the input."""
        (a,) = ctx.saved_values
        return wrap_tuple(operators.log_back(a, d_output))


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of two inputs."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient with respect to the inputs."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of the input."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient with respect to the input."""
        (a,) = ctx.saved_values
        return wrap_tuple(operators.inv_back(a, d_output))


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the negation of the input."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient with respect to the input."""
        return wrap_tuple(operators.neg(d_output))


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of the input."""
        sigmoid_value = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_value)
        return sigmoid_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient with respect to the input."""
        (sigmoid_value,) = ctx.saved_values
        return wrap_tuple(d_output * sigmoid_value * (1.0 - sigmoid_value))


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of the input."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient with respect to the input."""
        (a,) = ctx.saved_values
        return wrap_tuple(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    r"""Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of the input."""
        exp_value = operators.exp(a)
        ctx.save_for_backward(exp_value)
        return exp_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient with respect to the input."""
        (exp_value,) = ctx.saved_values
        return wrap_tuple(operators.mul(d_output, exp_value))


class LT(ScalarFunction):
    r"""Less than function $f(x, y) = 1.0 \text{ if } x < y \text{ else } 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1.0 if a is less than b, else 0.0."""
        return 1.0 if operators.lt(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Gradient is zero since this is a comparison function."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    r"""Equality function $f(x, y) = 1.0 \text{ if } x == y \text{ else } 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1.0 if a equals b, else 0.0."""
        return 1.0 if operators.eq(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Gradient is zero since this is a comparison function."""
        return 0.0, 0.0

"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: The product of x and y
    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
        x (float): Input number

    Returns:
        float: The input number unchanged
    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: The sum of x and y
    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x (float): Input number

    Returns:
        float: The negation of x
    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        bool: True if x < y, False otherwise
    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        bool: True if x == y, False otherwise
    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        float: The larger of x and y
    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close to each other.

    Args:
        x (float): First number
        y (float): Second number

    Returns:
        bool: True if |x - y| < 1e-2, False otherwise
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
        x (float): Input number

    Returns:
        float: The sigmoid of x
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function.

    Args:
        x (float): Input number

    Returns:
        float: max(0, x)
    """
    return max(0.0, x)


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
        x (float): Input number (must be positive)

    Returns:
        float: The natural logarithm of x
    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
        x (float): Input number

    Returns:
        float: e^x
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of log(x) * d.

    Args:
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
        float: The gradient of log(x) * d
    """
    return d / x


def inv(x: float) -> float:
    """Compute the inverse of x.

    Args:
        x (float): Input number (must be non-zero)

    Returns:
        float: 1/x
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of (1/x) * d.

    Args:
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
        float: The gradient of (1/x) * d
    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of ReLU(x) * d.

    Args:
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
        float: The gradient of ReLU(x) * d
    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element in a list.

    Args:
        fn (Callable[[float], float]): Function to apply
        ls (Iterable[float]): Input list

    Returns:
        Iterable[float]: List with fn applied to each element
    """
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Apply a function to pairs of elements from two lists.

    Args:
        fn (Callable[[float, float], float]): Function to apply
        ls1 (Iterable[float]): First input list
        ls2 (Iterable[float]): Second input list

    Returns:
        Iterable[float]: List with fn applied to pairs of elements
    """
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    fn: Callable[[float, float], float], start: float, ls: Iterable[float]
) -> float:
    """Reduce a list to a single value by applying a function cumulatively.

    Args:
        fn: The function to apply
        start: The initial value
        ls: The input list

    Returns:
        The final accumulated value
    """
    result = start
    for x in ls:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
        ls (Iterable[float]): Input list

    Returns:
        Iterable[float]: List with all elements negated
    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise.

    Args:
        ls1 (Iterable[float]): First input list
        ls2 (Iterable[float]): Second input list

    Returns:
        Iterable[float]: List with element-wise sum of inputs
    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
        ls (Iterable[float]): Input list

    Returns:
        float: Sum of all elements
    """
    return reduce(add, 0.0, ls)


def prod(ls: Iterable[float]) -> float:
    """Compute the product of all elements in a list.

    Args:
        ls (Iterable[float]): Input list

    Returns:
        float: Product of all elements
    """
    return reduce(mul, 1.0, ls)

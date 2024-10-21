from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol, Deque, Dict, Set


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals)

    # Slightly modify the value of the chosen argument by epsilon in both directions
    vals_list[arg] += epsilon
    f_x_plus = f(*vals_list)  # f at x + epsilon

    vals_list[arg] -= 2 * epsilon
    f_x_minus = f(*vals_list)  # f at x - epsilon

    # Compute central difference
    return (f_x_plus - f_x_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for the variable.

        Args:
            x: The derivative value to accumulate.
        """
        ...

    @property
    def unique_id(self) -> int:
        """Return a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Return True if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Return True if the variable is constant and does not require gradients."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute the derivatives for parent variables.

        Args:
            d_output: The derivative of the output with respect to this variable.

        Returns:
            An iterable of tuples containing parent variables and their corresponding derivatives.
        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    """
    grads: Dict[int, int] = defaultdict(int)
    grads[variable.unique_id] = 0

    stack: Deque[Variable] = deque([variable])
    visited: Set[int] = set([variable.unique_id])
    result: List[Variable] = []

    while stack:
        current_node = stack.pop()
        for node in current_node.parents:
            if not node.is_constant():
                grads[node.unique_id] += 1

                if node.unique_id not in visited:
                    stack.append(node)
                    visited.add(node.unique_id)

    stack.append(variable)
    while stack:
        cur_node = stack.pop()
        result.append(cur_node)

        for node in cur_node.parents:
            if not node.is_constant():
                grads[node.unique_id] -= 1
                if grads[node.unique_id] == 0:
                    stack.append(node)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    derivatives_dict = {variable.unique_id: deriv}

    top_sort = topological_sort(variable)

    # Iterate through the topological order and calculate the derivatives
    for curr_var in top_sort:
        if curr_var.is_leaf():
            continue

        # Get the derivatives of the current variable
        var_n_der = curr_var.chain_rule(derivatives_dict[curr_var.unique_id])

        # Accumulate the derivative for each parent of the current variable
        for var, deriv in var_n_der:
            if var.is_leaf():
                var.accumulate_derivative(deriv)
            else:
                if var.unique_id not in derivatives_dict:
                    derivatives_dict[var.unique_id] = deriv
                else:
                    derivatives_dict[var.unique_id] += deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values for backpropagation."""
        return self.saved_values

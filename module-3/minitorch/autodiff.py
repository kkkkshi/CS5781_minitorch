from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] += epsilon
    vals2[arg] -= epsilon
    return (f(*vals1) - f(*vals2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    Visited = set()
    result: List[Variable] = []

    def visit(n: Variable) -> None:
        if n.is_constant() or n.unique_id in Visited:
            return
        if not n.is_leaf():
            for input in n.parents:
                if not input.is_constant():
                    visit(input)
        Visited.add(n.unique_id)
        result.insert(0, n)

    visit(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # Call topological sort to get an ordered queue
    order = topological_sort(variable)
    # Create a dictionary of Scalars and current derivatives
    derivs = {variable.unique_id: deriv}
    # For each node in backward order
    for n in order:
        d_out = derivs[n.unique_id]
        #  if the Scalar is a leaf, add its final derivative (accumulate_derivative) and loop
        if n.is_leaf():
            n.accumulate_derivative(d_out)
        else:
            # call .chain_rule on the last function with d_output
            # loop through all the Scalars+derivative produced by the chain rule
            for key, item in n.chain_rule(d_out):
                if key.is_constant():
                    continue
                derivs.setdefault(key.unique_id, 0.0)
                derivs[key.unique_id] = derivs[key.unique_id] + item


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

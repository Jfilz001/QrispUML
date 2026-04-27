"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from numba import njit, prange

from qrisp.algorithms.qaoa import QAOAProblem, RX_mixer
from qrisp.core import rzz
from qrisp.jasp import check_for_tracing_mode

if TYPE_CHECKING:
    import networkx as nx

    from qrisp.core import QuantumVariable


def maxcut_obj(x: str, G: nx.Graph) -> int:
    """Compute the MaxCut objective value for a single bitstring and graph G."""
    return maxcut_obj_jitted(int(x[::-1], 2), list(G.edges()))


@njit(cache=True)
def maxcut_obj_jitted(x: int, edge_list: list[tuple[int, int]] | np.ndarray) -> int:
    """JIT-compiled MaxCut objective: count edges crossing the cut encoded by integer x."""
    cut = 0
    for i, j in edge_list:
        # the edge is cut
        if ((x >> i) ^ (x >> j)) & 1:
            # if x[i] != x[j]:
            cut -= 1
    return cut


@njit(parallel=True, cache=True)
def maxcut_energy(
    outcome_array: np.ndarray,
    count_array: np.ndarray,
    edge_list: np.ndarray,
) -> float:
    """Compute the weighted sum of MaxCut objectives over all measurement outcomes."""
    res_array = np.zeros(len(outcome_array))
    for i in prange(len(outcome_array)):
        res_array[i] = maxcut_obj_jitted(outcome_array[i], edge_list) * count_array[i]

    return np.sum(res_array)


def create_maxcut_cl_cost_function(G: nx.Graph) -> Callable[[dict], float]:
    """
    Creates the classical cost function for an instance of the maximum cut
    problem for a given graph ``G``.

    Parameters
    ----------
    G : nx.Graph
        The Graph for the problem instance.

    Returns
    -------
    cl_cost_function : Callable[[dict], float]
        The classical cost function for the problem instance, which takes a
        dictionary of measurement results as input.

    """

    def cl_cost_function(counts: dict) -> float:

        edge_list = np.array(list(G.edges()), dtype=np.uint32)

        counts_keys = list(counts.keys())

        int_list = []
        if not isinstance(counts_keys[0], str):

            for c_array in counts_keys:
                integer = int("".join(list(c_array))[::-1], 2)
                int_list.append(integer)
        else:
            for c_str in counts_keys:
                integer = int(c_str[::-1], 2)
                int_list.append(integer)

        counts_array = np.array(list(counts.values()))
        outcome_array = np.array(int_list, dtype=np.uint32)

        return maxcut_energy(outcome_array, counts_array, edge_list)

    return cl_cost_function


@jit
def extract_boolean_digit(integer: Any, digit: Any) -> Any:
    """Extract a single boolean digit from an integer at the given bit position."""
    return (integer >> digit) & 1


def create_cut_computer(G: nx.Graph) -> Callable[[int], jnp.ndarray]:
    """
    Create a JIT-compiled function that computes the cut value for a given graph G.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    cut_computer : Callable[[int], jnp.ndarray]
        A JIT-compiled function mapping an integer bitstring to its (negated) cut value.

    """
    edge_list = jnp.array(G.edges())

    @jit
    def cut_computer(x: int) -> jnp.ndarray:
        x_uint = jnp.uint32(x)
        bools = extract_boolean_digit(x_uint, edge_list[:, 0]) != extract_boolean_digit(
            x_uint, edge_list[:, 1]
        )
        # Count the number of edges crossing the cut
        cut = jnp.sum(bools)
        return -cut

    return cut_computer


def create_maxcut_sample_array_post_processor(G: nx.Graph) -> Callable[[Any], Any]:
    """
    Create a post-processor that computes the average cut value over a sample array.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    post_processor : Callable[[Any], Any]
        A function that takes a sample array and returns the mean cut value.

    """
    cut_computer = create_cut_computer(G)

    def post_processor(sample_array: Any) -> Any:
        # Use vmap for automatic vectorization
        cut_values = vmap(cut_computer)(sample_array)
        average_cut = jnp.mean(cut_values)
        return average_cut

    return post_processor


def create_maxcut_cost_operator(
    G: nx.Graph,
) -> Callable[[QuantumVariable, float], None]:
    r"""
    Creates the cost operator for an instance of the maximum cut problem for a given graph ``G``.

    Parameters
    ----------
    G : nx.Graph
        The Graph for the problem instance.

    Returns
    -------
    maxcut_cost_operator : Callable[[QuantumVariable, float], None]
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$.
        This function performs the application of the cost operator.

    """

    def maxcut_cost_operator(qv: QuantumVariable, gamma: float) -> None:

        if not check_for_tracing_mode():
            if len(G) != len(qv):
                raise ValueError(
                    f"Tried to call MaxCut cost Operator for graph of size "
                    f"{len(G)} on argument of invalid size {len(qv)}"
                )

        for pair in list(G.edges()):
            rzz(2 * gamma, qv[pair[0]], qv[pair[1]])
            # cx(qv[pair[0]], qv[pair[1]])
            # rz(2 * gamma, qv[pair[1]])
            # cx(qv[pair[0]], qv[pair[1]])
            # barrier(qv)

    return maxcut_cost_operator


def maxcut_problem(G: nx.Graph) -> QAOAProblem:
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for MaxCut for a given graph ``G``.

    """

    return QAOAProblem(
        create_maxcut_cost_operator(G), RX_mixer, create_maxcut_cl_cost_function(G)
    )

# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, cast

import torch
from numpy import ndarray
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen
from torch import Tensor, float32, nn
from torch.distributions import Distribution, Uniform

from sbi.sbi_types import Array
from sbi.utils.sbiutils import within_support
from sbi.utils.torchutils import BoxUniform, atleast_2d
from sbi.utils.user_input_checks_utils import (
    CustomPriorWrapper,
    MultipleIndependent,
    OneDimPriorWrapper,
    PytorchReturnTypeWrapper,
)


def check_prior(prior: Any) -> None:
    """Assert that prior is a PyTorch distribution (or pass if None)."""

    if prior is None:
        pass
    else:
        assert isinstance(
            prior, Distribution
        ), """Prior must be a PyTorch Distribution. See FAQ 7 for more details or use
        `sbi.utils.user_input_checks.process_prior` for wrapping scipy and lists of
        independent priors."""


def process_prior(
    prior: Union[Sequence[Distribution], Distribution, rv_frozen, multi_rv_frozen],
    custom_prior_wrapper_kwargs: Optional[Dict] = None,
) -> Tuple[Distribution, int, bool]:
    """
    Return PyTorch distribution-like prior from user-provided prior.

    NOTE: If the prior argument is a sequence of PyTorch distributions, \
    they will be interpreted as independent prior dimensions wrapped in a \
    :class:`MultipleIndependent` PyTorch Distribution. In case the elements \
    are not PyTorch distributions, make sure to use :func:`process_prior` \
    on each element in the list beforehand.

    NOTE: Returns a tuple `(processed_prior, num_params, \
        whether_prior_returns_numpy)`. The last two entries in the tuple\
        can be passed on to `process_simulator` to prepare the simulator. For \
        example, it ensures parameters are cast to numpy or adds a batch \
        dimension to the simulator output, if needed.

    Args:
        prior (:class:`~torch.distributions.distribution.Distribution` \
            or Sequence[:class:`~torch.distributions.distribution.Distribution`]):
            Prior object with `.sample()` and `.log_prob()`, or a sequence \
                of such objects.
        custom_prior_wrapper_kwargs (dict, optional):
            Additional arguments passed to the wrapper class that processes the prior
            into a PyTorch Distribution, such as bounds (`lower_bound`, `upper_bound`)
            or argument constraints (`arg_constraints`).

    Raises:
        AttributeError: If prior objects lack `.sample()` or `.log_prob()`.

    Returns:
        Tuple[torch.distributions.Distribution, int, bool]:
            - `prior`: A PyTorch-compatible prior.
            - `theta_numel`: Dimensionality of a single sample from the prior.
            - `prior_returns_numpy`: Whether the prior originally returned NumPy arrays.

    Example:
    --------

    ::

        import torch
        from torch.distributions import Uniform
        from sbi.utils.user_input_checks import process_prior

        prior = Uniform(torch.zeros(1), torch.ones(1))
        prior, theta_numel, prior_returns_numpy = process_prior(prior)
    """

    # If prior is a sequence, assume independent components and check as PyTorch prior.
    if isinstance(prior, Sequence):
        warnings.warn(
            f"Prior was provided as a sequence of {len(prior)} priors. They will be "
            "interpreted as independent of each other and matched in order to the "
            "components of the parameter.",
            stacklevel=2,
        )
        # process individual priors
        prior = [process_prior(p, custom_prior_wrapper_kwargs)[0] for p in prior]
        return process_pytorch_prior(MultipleIndependent(prior))

    if isinstance(prior, Distribution):
        return process_pytorch_prior(prior)

    # If prior is given as `scipy.stats` object, wrap as PyTorch.
    elif isinstance(prior, (rv_frozen, multi_rv_frozen)):
        raise NotImplementedError(
            "Passing a prior as scipy.stats object is deprecated. "
            "Please pass it as a PyTorch Distribution."
        )

    # Otherwise it is a custom prior - check for `.sample()` and `.log_prob()`.
    else:
        return process_custom_prior(prior, custom_prior_wrapper_kwargs)


def process_custom_prior(
    prior, custom_prior_wrapper_kwargs: Optional[Dict] = None
) -> Tuple[Distribution, int, bool]:
    """Check and return corrected prior object defined by the user.

    Args:
        prior: Prior object with `.sample()` and `.log_prob()` as provided by the user.
        custom_prior_wrapper_kwargs: kwargs to be passed to the class that wraps a
            custom prior into a pytorch Distribution, e.g., for passing bounds for a
            prior with bounded support (lower_bound, upper_bound), or argument
            constraints.
            (arg_constraints), see pytorch.distributions.Distribution for more info.

    Returns:
        prior: sbi-compatible prior.
        theta_numel: Number of parameters - elements in a single sample from the prior.
        is_prior_numpy: Whether the prior returned Numpy arrays before wrapping.
    """

    check_prior_attributes(prior)
    check_prior_batch_behavior(prior)
    prior, is_prior_numpy = maybe_wrap_prior_as_pytorch(
        prior, custom_prior_wrapper_kwargs
    )
    check_prior_return_type(prior)
    theta_numel = prior.sample().numel()

    return prior, theta_numel, is_prior_numpy


def maybe_wrap_prior_as_pytorch(
    prior, custom_prior_wrapper_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Distribution, bool]:
    """Check prior return type and maybe wrap as PyTorch.

    Args:
        prior: Prior object with `.sample()` and `.log_prob()` as provided by the user.
        custom_prior_wrapper_kwargs: kwargs to be passed to the class that wraps a
            custom prior into a pytorch Distribution, e.g., for passing bounds for a
            prior with bounded support (lower_bound, upper_bound), or argument
            constraints.
            (arg_constraints), see pytorch.distributions.Distribution for more info.

    Raises:
        TypeError: If prior return type is PyTorch or Numpy.

    Returns:
        prior: Prior that emits samples and evaluates log prob as PyTorch Tensors.
        is_prior_numpy: Whether the prior returned Numpy arrays before wrapping.
    """

    theta = prior.sample((1,))
    log_probs = prior.log_prob(theta)

    # Check return types
    if isinstance(theta, Tensor) and isinstance(log_probs, Tensor):
        # XXX: We wrap to get a Distribution. But this might interfere with the fact
        # that the custom prior can be a probabilistic program.
        prior = CustomPriorWrapper(
            custom_prior=prior,
            event_shape=torch.Size([theta.numel()]),
            **custom_prior_wrapper_kwargs or {},
        )
        is_prior_numpy = False
    elif isinstance(theta, ndarray) and isinstance(log_probs, ndarray):
        # infer event shape from single numpy sample.
        event_shape = torch.Size([theta.size])
        prior = CustomPriorWrapper(
            custom_prior=prior,
            event_shape=event_shape,
            **custom_prior_wrapper_kwargs or {},
        )
        is_prior_numpy = True
    else:
        raise TypeError(
            f"Prior must return torch.Tensor or ndarray, but returns {type(theta)}"
        )
    return cast(Distribution, prior), is_prior_numpy


def process_pytorch_prior(prior: Distribution) -> Tuple[Distribution, int, bool]:
    """Return PyTorch prior adapted to the requirements for sbi.

    Args:
        prior: PyTorch distribution prior provided by the user.

    Raises:
        ValueError: If prior is defined over an unwrapped scalar variable.

    Returns:
        prior: PyTorch distribution prior.
        theta_numel: Number of parameters - elements in a single sample from the prior.
        prior_returns_numpy: False.
    """

    # Turn off validation of input arguments to allow `log_prob()` on samples outside
    # of the support.
    prior.set_default_validate_args(False)

    # Reject unwrapped scalar priors.
    # This will reject Uniform priors with dimension larger than 1.
    if prior.sample().ndim == 0:
        raise ValueError(
            "Detected scalar prior. Please make sure to pass a PyTorch prior with "
            "`batch_shape=torch.Size([1])` or `event_shape=torch.Size([1])`."
        )
    # Cast 1D Uniform to BoxUniform to avoid shape error in mdn log prob.
    elif isinstance(prior, Uniform) and prior.batch_shape.numel() == 1:
        prior = BoxUniform(low=prior.low, high=prior.high)
        warnings.warn(
            "Casting 1D Uniform prior to BoxUniform to match sbi batch requirements.",
            stacklevel=2,
        )

    check_prior_batch_behavior(prior)
    check_prior_batch_dims(prior)

    if prior.sample().dtype != float32:
        prior = PytorchReturnTypeWrapper(
            prior, return_type=float32, validate_args=False
        )

    # This will fail for float64 priors.
    check_prior_return_type(prior)

    # Potentially required wrapper if the prior returns an additional sample dimension
    # for `.log_prob()`.
    if prior.log_prob(prior.sample(torch.Size((10,)))).shape == torch.Size([10, 1]):
        prior = OneDimPriorWrapper(prior, validate_args=False)

    theta_numel = prior.sample().numel()

    return prior, theta_numel, False


def check_prior_batch_dims(prior) -> None:
    """Check if batch shape of the prior is smaller or equal to 1.

    Raises:
        ValueError: If `batch_shape` larger than 1.
    """

    if prior.batch_shape.numel() > 1:
        raise ValueError(
            """The specified prior has `batch_shape` larger than 1. Please
            specify a prior with batch_shape smaller equal to 1 and `event_shape`
            equal to number of parameters of your model.

            In case your intention was to pass a univariate distribution like Uniform
            (or Beta, Gamma, etc.) defined over multiple parameters, consider instead
            using `torch.distributions.Independent` to reinterpret batch dimensions as
            event dimensions, or use the `MultipleIndependent` distribution we provide.

            To use `sbi.utils.MultipleIndependent`, just pass a list of priors, e.g. to
            specify a uniform prior over two parameters, pass as prior:
                prior = [
                            Uniform(torch.zeros(1), torch.ones(1)),
                            Uniform(torch.ones(1), 2 * torch.ones(1))
                        ]
            or, to pass a Gamma over the first parameter and a correlated Gaussian over
            the other two, write:
                prior = [
                            Gamma(torch.ones(1), 2 * torch.ones(1)),
                            MVG(torch.zeros(2), tensor([[1., 0.1], [0.1, 2.]])),
                        ]
            """
        )


def check_for_possibly_batched_x_shape(x_shape):
    """Raise `ValueError` if dimensionality of simulations doesn't match requirements.

    sbi does not support multiple observations yet. For 2D observed data the leading
    dimension will be interpreted as batch dimension and a `ValueError` is raised if the
    batch dimension is larger than 1.

    Multidimensional observations e.g., images, are allowed when they are passed with an
    additional leading batch dimension of size 1.
    """

    x_ndim = len(x_shape)
    inferred_batch_shape, *inferred_data_shape = x_shape

    # Reject multidimensional data with batch_shape > 1.
    if x_ndim > 1 and inferred_batch_shape > 1:
        raise ValueError(
            """The `x` passed to condition the posterior for evaluation or sampling
                has an inferred batch shape larger than one. This is not supported in
                some sbi methods for reasons depending on the scenario:

                    - in case you want to evaluate or sample conditioned on several iid
                      xs e.g., (p(theta | [x1, x2, x3])), this is fully supported only
                      for likelihood based SNLE and SNRE. For SNPE it is supported only
                      for a fixed number of trials and using an appropriate embedding
                      net, i.e., by treating the trials as additional data dimension. In
                      that case, make sure to pass xo with a leading batch dimensionen.

                    - in case you trained with a single round to do amortized inference
                    and now you want to evaluate or sample a given theta conditioned on
                    several xs, one after the other, e.g, p(theta | x1), p(theta | x2),
                    p(theta| x3): this broadcasting across xs is not supported in sbi.
                    Instead, what you can do it to call posterior.log_prob(theta, xi)
                    multiple times with different xi.

                    - finally, if your observation is multidimensional, e.g., an image,
                    make sure to pass it with a leading batch dimension, e.g., with
                    shape (1, xdim1, xdim2). Beware that the current implementation
                    of sbi might not provide stable support for this and result in
                    shape mismatches.

            NOTE: below we use list notation to reduce clutter, but `x` should be of
            type torch.Tensor or ndarray.

            For example:

            > x_o = [[1]]
            > x_o = [[1, 2, 3]]

            are interpreted as single observations with a leading batch dimension of
            one. However

            > x_o = [ [1], [2] ]
            > x_o = [ [1,2,3], [4,5,6] ]

            are interpreted as a batch of two scalar or vector observations, which
            is not supported yet. The following is interpreted as a matrix-shaped
            observation, e.g. a monochromatic image:

            > x_o = [ [[1,2,3], [4,5,6]] ]

            Finally, for convenience,

            > x_o = [1]
            > x_o = [1, 2, 3]

            will be interpreted as a single scalar or single vector observation
            respectively, without the user needing to wrap or unsqueeze them.
            """
        )
    else:
        pass


def check_prior_attributes(prior) -> None:
    """Check for prior methods sample(sample_shape) .log_prob(value) methods.

    Raises:
        AttributeError: if either of the two methods doesn't exist.
    """

    # Sample a batch of two parameters to check batch behaviour > 1 and that
    # `.sample()` can handle a tuple argument.
    num_samples = 2
    try:
        theta = prior.sample((num_samples,))
    except AttributeError as err:
        raise AttributeError(
            "Prior needs method `.sample()`. Consider using a PyTorch distribution."
        ) from err
    except TypeError as err:
        raise TypeError(
            f"""The `prior.sample()` method must accept Tuple arguments, e.g.,
            prior.sample(({num_samples}, )) to sample a batch of 2 parameters. Consider
            using a PyTorch distribution."""
        ) from err
    except Exception as err:  # Catch any other error.
        raise ValueError(
            f"""Something went wrong when sampling a batch of parameters
            from the prior as `prior.sample(({num_samples}, ))`. Consider using a
            PyTorch distribution."""
        ) from err
    try:
        prior.log_prob(theta)
    except AttributeError as err:
        raise AttributeError(
            "Prior needs method `.log_prob()`. Consider using a PyTorch distribution."
        ) from err
    except Exception as err:  # Catch any other error.
        raise ValueError(
            """Something went wrong when evaluating a batch of parameters theta
            with `prior.log_prob(theta)`. Consider using a PyTorch distribution."""
        ) from err


def check_prior_return_type(
    prior, return_type: Optional[torch.dtype] = float32
) -> None:
    """Check whether prior.sample() returns float32 Tensor."""

    prior_dtype = prior.sample().dtype
    assert prior_dtype == return_type, (
        f"Prior return type must be {return_type}, but is {prior_dtype}."
    )


def check_prior_batch_behavior(prior) -> None:
    """Assert that it is possible to sample and evaluate batches of parameters."""

    # Check for correct batch size in .sample and .log_prob
    num_samples = 1
    theta = prior.sample((num_samples,))
    log_probs = prior.log_prob(theta)

    assert (
        len(theta.shape) >= 2
    ), f"""A parameter batch sampled from the prior must be at least 2D,
    (num_samples, parameter_dim), but is {len(theta.shape)}"""

    num_sampled, *parameter_dim = theta.shape
    # Using len here because `log_prob` could be `ndarray` or `torch.Tensor`.
    num_log_probs = len(log_probs)

    assert num_sampled == num_samples, (
        "prior.sample((batch_size, )) must return batch_size parameters."
    )

    assert num_log_probs == num_samples, (
        "prior.log_prob must return as many log probs as samples."
    )


def check_prior_support(prior):
    """Check whether prior allows to check for support.

    This either uses the PyTorch support property, or the custom prior .logprob method
    """

    try:
        within_support(prior, prior.sample((1,)))
    except NotImplementedError as err:
        raise NotImplementedError(
            """The prior must implement the support property or allow to call
            .log_prob() outside of support."""
        ) from err


def check_data_device(datum_1: torch.Tensor, datum_2: torch.Tensor) -> None:
    """Checks if two tensors have the seme device. Fails if there is a device
    discrepancy

    Args:
        datum_1: torch `Tensor`
        datum_2: torch `Tensor`
    """
    assert datum_1.device == datum_2.device, (
        "Mismatch in fed data's device: "
        f"datum_1 has device '{datum_1.device}' whereas "
        f"datum_2 has device '{datum_2.device}'. Please "
        "use data from a common device."
    )


def process_simulator(
    user_simulator: Callable,
    prior: Distribution,
    is_numpy_simulator: bool,
) -> Callable:
    """Returns a simulator that meets the requirements for usage in sbi.

    Args:
        user_simulator (Callable):
            simulator provided by the user, possibly written in numpy.
        prior (torch.distributions.Distribution):
            prior as pytorch distribution or processed with :func:`process_prior`.
        is_numpy_simulator (bool):
            whether the simulator needs theta in numpy types, returned
            from `process_prior`.

    Returns:
        Callable:
            simulator: processed simulator that returns :class:`torch.Tensor` \
                and can handle batches of parameters.

    Example:
    --------

    ::

        import torch
        from sbi.utils.user_input_checks import process_simulator
        from torch.distributions import Uniform
        from sbi.utils.user_input_checks import process_prior

        prior = Uniform(torch.zeros(1), torch.ones(1))
        prior, theta_numel, prior_returns_numpy = process_prior(prior)
        simulator = lambda theta: theta + 1
        simulator = process_simulator(simulator, prior, prior_returns_numpy)
    """

    assert isinstance(user_simulator, Callable), "Simulator must be a function."

    joblib_simulator = wrap_as_joblib_efficient_simulator(
        user_simulator, prior, is_numpy_simulator
    )

    batch_simulator = ensure_batched_simulator(joblib_simulator, prior)

    return batch_simulator


# New simulator wrapper, deriving from #1175 refactoring of simulate_for_sbi.
# For now it just blindly applies a cast to tensor to the input and the output
# of the simulator. This is not efficient (~3 times slowdown), but is compatible
# with the new joblib and, importantly, does not break previous code. It should
# be removed with a future restructuring of the simulation pipeline.
def wrap_as_joblib_efficient_simulator(
    simulator: Callable, prior, is_numpy_simulator
) -> Callable:
    """Return a simulator that accepts `ndarray` and returns `Tensor` arguments."""

    def joblib_simulator(theta: ndarray) -> Tensor:
        return torch.as_tensor(simulator(torch.as_tensor(theta)), dtype=float32)

    return joblib_simulator


# Probably not used anymore
def wrap_as_pytorch_simulator(
    simulator: Callable, prior, is_numpy_simulator
) -> Callable:
    """Return a simulator that accepts and returns `Tensor` arguments."""

    if is_numpy_simulator:
        # Get data to check input type is consistent with data.
        theta = prior.sample().numpy()  # Cast to numpy because is in PyTorch already.
        x = simulator(theta)
        assert isinstance(x, ndarray), (
            f"Simulator output type {type(x)} must match its input type {type(theta)}"
        )

        # Define a wrapper function to PyTorch
        def pytorch_simulator(theta: Tensor) -> Tensor:
            return torch.as_tensor(simulator(theta.numpy()), dtype=float32)

    else:
        # Define a wrapper to make sure that the output of the simulator is `float32`.
        def pytorch_simulator(theta: Tensor) -> Tensor:
            return torch.as_tensor(simulator(theta), dtype=float32)

    return pytorch_simulator


def ensure_batched_simulator(simulator: Callable, prior) -> Callable:
    """Return a simulator with batched output.

    Return the unchanged simulator if it can already simulate multiple parameter
    vectors per call. Otherwise, wrap as simulator with batched output (leading batch
    dimension of shape [1]).
    """

    is_batched_simulator = True
    try:
        batch_size = 2
        # The simulator must return a matching batch dimension and data.
        output_shape = simulator(prior.sample((batch_size,))).shape
        assert len(output_shape) > 1
        assert output_shape[0] == batch_size
    except Exception:
        is_batched_simulator = False

    return simulator if is_batched_simulator else get_batch_loop_simulator(simulator)


def get_batch_loop_simulator(simulator: Callable) -> Callable:
    """Return simulator wrapped with `map` to handle batches of parameters.

    Note: this batches the simulator only syntactically, there are no performance
    benefits as with true vectorization."""

    def batch_loop_simulator(theta: Tensor) -> Tensor:
        """Return a batch of simulations by looping over a batch of parameters."""
        assert theta.ndim > 1, "Theta must have a batch dimension."
        # Simulate in loop
        xs = list(map(simulator, theta))
        # Stack over batch to keep x_shape
        return torch.stack(xs)

    return batch_loop_simulator


def process_x(x: Array, x_event_shape: Optional[torch.Size] = None) -> Tensor:
    """Return observed data adapted to match sbi's shape and type requirements.

    This means that `x` is returned with a `batch_dim`.

    If `x_shape` is `None`, the shape is not checked.

    Args:
        x: Observed data as provided by the user.
        x_event_shape: Prescribed shape - either directly provided by the user at init
            or inferred by sbi by running a simulation and checking the output. Does not
            contain a batch dimension.

    Returns:
        x: Observed data with shape ready for usage in sbi.
    """

    x = atleast_2d(torch.as_tensor(x, dtype=float32))

    if x_event_shape is not None and len(x_event_shape) > len(x.shape):
        raise ValueError(
            f"You passed an `x` of shape {x.shape} but the `x_event_shape` (inferred "
            f"from simulations) is {x_event_shape}. We are raising this error because "
            f"len(x_event_shape) > len(x.shape)"
        )

    # If x_shape is provided, we can fix a missing batch dim for >1D data.
    if x_event_shape is not None and len(x_event_shape) == len(x.shape):
        x = x.unsqueeze(0)

    input_x_shape = x.shape

    if x_event_shape is not None:
        # Number of trials can change for every new x, but single trial x shape must
        # match.
        assert input_x_shape[1:] == x_event_shape, (
            f"Observed data shape ({input_x_shape[1:]}) must match "
            f"the shape of simulated data x ({x_event_shape})."
        )
    return x


def prepare_for_sbi(simulator: Callable, prior) -> Tuple[Callable, Distribution]:
    """Prepare simulator and prior for usage in sbi.

    NOTE: This method is deprecated as of sbi version v0.23.0. and will be removed in a
    future release. Please use `process_prior` and `process_simulator` in the future.
    This is a wrapper around `process_prior` and `process_simulator` which can be
    used in isolation as well.

    Attempts to meet the following requirements by reshaping and type-casting:

    - the simulator function receives as input and returns a Tensor.<br/>
    - the simulator can simulate batches of parameters and return batches of data.<br/>
    - the prior does not produce batches and samples and evaluates to Tensor.<br/>
    - the output shape is a `torch.Size((1,N))` (i.e, has a leading batch dimension 1).

    If this is not possible, a suitable exception will be raised.

    Args:
        simulator: Simulator as provided by the user.
        prior: Prior as provided by the user.

    Returns:
        Tuple (simulator, prior) checked and matching the requirements of sbi.
    """

    warnings.warn(
        "This method is deprecated as of sbi version v0.23.0. and will be removed in a \
        future release."
        "Please use `process_prior` and `process_simulator` in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Check prior, return PyTorch prior.
    prior, _, prior_returns_numpy = process_prior(prior)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    # Consistency check after making ready for sbi.
    check_sbi_inputs(simulator, prior)

    return simulator, prior


def check_sbi_inputs(simulator: Callable, prior: Distribution) -> None:
    """Assert requirements for simulator, prior and observation for usage in sbi.

    Args:
        simulator: simulator function
        prior: prior (Distribution like)
        x_shape: Shape of single simulation output $x$.
    """
    check_prior_support(prior)
    num_prior_samples = 1
    theta = prior.sample(torch.Size((num_prior_samples,)))
    theta_batch_shape, *_ = theta.shape
    simulation = simulator(theta)
    sim_batch_shape, *sim_event_shape = simulation.shape

    assert isinstance(theta, Tensor), "Parameters theta must be a `Tensor`."
    assert isinstance(simulation, Tensor), "Simulator output must be a `Tensor`."

    assert (
        theta_batch_shape == num_prior_samples
    ), f"""Theta batch shape {theta_batch_shape} must match
        num_samples={num_prior_samples}."""
    assert (
        sim_batch_shape == num_prior_samples
    ), f"""Simulation batch shape {sim_batch_shape} must match
        num_samples={num_prior_samples}."""


def check_estimator_arg(estimator: Union[str, Callable]) -> None:
    """Check (density or ratio) estimator argument passed by the user."""
    assert isinstance(estimator, str) or (
        isinstance(estimator, Callable) and not isinstance(estimator, nn.Module)
    ), (
        "The passed density estimator / classifier must be a string or a function "
        f"returning a nn.Module, but is {type(estimator)}"
    )


def validate_theta_and_x(
    theta: Any, x: Any, data_device: str = "cpu", training_device: str = "cpu"
) -> Tuple[Tensor, Tensor]:
    r"""
    Checks if the passed $(\theta, x)$ are valid.

    Specifically, we check:
    1) If they are (torch) tensors.
    2) If they have the same batchsize.
    3) If they are of `dtype=float32`.

    Additionally, We move the data to the specified `data_device`. This is where the
    data is stored and can be separate from `training_device`, where the
    computations for training are performed.

    Raises:
        AssertionError: If theta or x are not torch.Tensor-like,
        do not yield the same batchsize and do not have dtype==float32.

    Args:
        theta: Parameters.
        x: Simulation outputs.
        data_device: Device where data is stored.
        training_device: Training device for net.
    """
    assert isinstance(theta, Tensor), "Parameters theta must be a `torch.Tensor`."
    assert isinstance(x, Tensor), "Simulator output must be a `torch.Tensor`."

    assert theta.shape[0] == x.shape[0], (
        f"Number of parameter sets (={theta.shape[0]} must match the number of "
        f"simulation outputs (={x.shape[0]})"
    )

    # I did not fuse these asserts with the `isinstance(x, Tensor)` asserts in order
    # to give more explicit errors.
    assert theta.dtype == float32, "Type of parameters must be float32."
    assert x.dtype == float32, "Type of simulator outputs must be float32."

    if str(x.device) != str(data_device):
        warnings.warn(
            f"Data x has device '{x.device}'. "
            f"Moving x to the data_device '{data_device}'. "
            f"Training will proceed on device '{training_device}'.",
            stacklevel=2,
        )
        x = x.to(data_device)

    if str(theta.device) != str(data_device):
        warnings.warn(
            f"Parameters theta has device '{theta.device}'. "
            f"Moving theta to the data_device '{data_device}'. "
            f"Training will proceed on device '{training_device}'.",
            stacklevel=2,
        )
        theta = theta.to(data_device)

    return theta, x


def test_posterior_net_for_multi_d_x(net, theta: Tensor, x: Tensor) -> None:
    """Test log prob method of the net.

    This is done to make sure the net can handle multidimensional inputs via an
    embedding net. If not, it usually fails with a RuntimeError. Here we catch the
    error, append a debug hint and raise it again.

    Args:
        net: A `DensityEstimator`.
    """
    try:
        # torch.nn.functional needs at least two inputs here.
        if hasattr(net, "log_prob"):
            # This only is checked for density estimators, not for classifiers and
            # others
            net.log_prob(theta[:, :2], condition=x[:2])

    except RuntimeError as rte:
        ndims = x.ndim
        if ndims > 2:
            message = f"""Debug hint: The simulated data x has {ndims - 1} dimensions.
            With default settings, sbi cannot deal with multidimensional simulations.
            Make sure to use an embedding net that reduces the dimensionality, e.g., a
            CNN in case of images, or change the simulator to return one-dimensional x.
            """
        else:
            message = ""

        raise RuntimeError(message) from rte

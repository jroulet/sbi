# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import xgboost
from scipy.optimize import differential_evolution
from torch import Tensor
from torch.distributions import Distribution
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

import sbi.utils as utils
from sbi.inference.trainers._contracts import LossArgsNPE
from sbi.inference.trainers.base import LossArgs
from sbi.inference.trainers.npe.npe_b import NPE_B
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.neural_nets.estimators.shape_handling import reshape_to_sample_batch_event
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries


class NPE_D(NPE_B):
    r"""Neural Posterior Estimation with counterweighted importance weights.

    NPE-D extends NPE-B by reducing the variance of the prior-to-proposal importance
    weights. For simulations from a proposal $\tilde p(\theta)$, NPE-B weights the
    negative log-likelihood by $p(\theta) / \tilde p(\theta)$. NPE-D additionally
    multiplies the weights by a positive function of the data only. This changes the
    measure over observations but leaves the optimum $q_\phi(\theta|x)=p(\theta|x)$
    unchanged.

    Following the counterweight method introduced in Roulet et al. (2026), two
    XGBoost regressors predict the conditional mean and dispersion of the log
    prior-to-proposal ratio. Their coefficients are selected by maximizing the
    effective sample size of the training weights. In sequential inference, the
    proposal is the sample-count-weighted mixture of all proposals used so far.

    The counterweight regressors are fitted whenever a new round is trained. They are
    fitted only on the neural density estimator's training split and then evaluated on
    both the training and validation splits.

    Reference:
        *A domain-optimized machine-learning tool for gravitational wave inference*,
        Roulet, Crisostomi, Thomas & Chatziioannou (2026).
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["nsf", "maf", "mdn", "made"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
        xgb_regressor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""Initialize NPE-D.

        Args:
            prior: Prior distribution over parameters.
            density_estimator: Conditional density estimator or estimator builder.
            device: Training device, e.g. ``"cpu"`` or ``"cuda"``.
            logging_level: Minimum severity of messages to log.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
            tracker: Tracking adapter used to log training metrics.
            show_progress_bars: Whether to show progress bars during training.
            xgb_regressor_kwargs: Optional keyword arguments passed to both
                ``xgboost.XGBRegressor`` instances.
        """

        self._xgb_regressor_kwargs = xgb_regressor_kwargs or {}
        self._counterweight_cache_key: Optional[Tuple[int, int]] = None
        self._counterweight_weights: Optional[Tensor] = None
        self._raw_importance_efficiency = 1.0
        self._counterweight_efficiency = 1.0

        kwargs = del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
                "xgb_regressor_kwargs",
            ),
        )
        super().__init__(**kwargs)

    @property
    def counterweight_efficiency(self) -> float:
        """Effective sample fraction of the latest counterweighted training set."""

        return self._counterweight_efficiency

    @property
    def raw_importance_efficiency(self) -> float:
        """Effective sample fraction before applying the latest counterweights."""

        return self._raw_importance_efficiency

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders containing counterweights as a fourth tensor."""

        theta, x, prior_masks = self.get_simulations(starting_round)
        num_examples = theta.size(0)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        if not resume_training:
            permuted_indices = torch.randperm(num_examples)
            self.train_indices, self.val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        cache_key = (starting_round, num_examples)
        if not resume_training or self._counterweight_cache_key != cache_key:
            self._counterweight_weights = self._fit_counterweights(theta, x)
            self._counterweight_cache_key = cache_key

        assert self._counterweight_weights is not None
        dataset = data.TensorDataset(theta, x, prior_masks, self._counterweight_weights)

        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        return (
            data.DataLoader(dataset, **train_loader_kwargs),
            data.DataLoader(dataset, **val_loader_kwargs),
        )

    def _fit_counterweights(self, theta: Tensor, x: Tensor) -> Tensor:
        """Fit counterweight regressors and return all sample weights."""

        log_prior_ratios = self._log_prior_to_proposal_ratio(theta)
        if self._round == 0:
            self._raw_importance_efficiency = 1.0
            self._counterweight_efficiency = 1.0
            return torch.ones(len(theta), dtype=theta.dtype, device=theta.device)

        x_numpy = x.detach().cpu().numpy().reshape(len(x), -1)
        # Keep XGBoost inputs in SBI's native float32 dtype. On macOS, fitting
        # consecutive XGBRegressor instances with float64 labels can terminate the
        # process inside XGBoost's NumPy metadata conversion.
        ratios_numpy = log_prior_ratios.detach().cpu().float().numpy()
        train_indices = self.train_indices.detach().cpu().numpy()
        x_train = x_numpy[train_indices]
        ratios_train = ratios_numpy[train_indices]

        booster_mu = xgboost.XGBRegressor(**self._xgb_regressor_kwargs)
        booster_mu.fit(x_train, ratios_train)
        mu_train = booster_mu.predict(x_train)

        booster_sigma = xgboost.XGBRegressor(**self._xgb_regressor_kwargs)
        booster_sigma.fit(x_train, np.sqrt(np.abs(ratios_train**2 - mu_train**2)))
        sigma_train = booster_sigma.predict(x_train)

        def negative_efficiency(coefficients: np.ndarray) -> float:
            weights = self._weights(ratios_train, mu_train, sigma_train, *coefficients)
            return -self._effective_sample_fraction(weights)

        result = differential_evolution(
            negative_efficiency,
            bounds=[(0.0, 2.0), (0.0, 4.0)],
        )
        coefficients = (float(result.x[0]), float(result.x[1]))

        mu = booster_mu.predict(x_numpy)
        sigma = booster_sigma.predict(x_numpy)
        weights = self._weights(ratios_numpy, mu, sigma, *coefficients)

        self._raw_importance_efficiency = self._effective_sample_fraction(
            np.exp(ratios_train)
        )
        self._counterweight_efficiency = self._effective_sample_fraction(
            weights[train_indices]
        )

        return torch.as_tensor(weights, dtype=theta.dtype, device=theta.device)

    def _log_prior_to_proposal_ratio(self, theta: Tensor) -> Tensor:
        """Evaluate the prior-to-mixture-proposal log density ratio."""

        log_prior = self._prior.log_prob(theta)
        utils.assert_not_nan_or_plus_inf(
            log_prior, "prior log probs of proposal samples"
        )

        num_samples_by_round = torch.tensor(
            [round_theta.size(0) for round_theta in self._theta_roundwise],
            dtype=theta.dtype,
            device=theta.device,
        )
        log_mixture_weights = torch.log(
            num_samples_by_round / num_samples_by_round.sum()
        )

        log_proposals = []
        for proposal in self._proposal_roundwise:
            density = self._prior if proposal is None else proposal
            log_density = density.log_prob(theta)
            utils.assert_not_nan_or_plus_inf(
                log_density, "proposal log probs of proposal samples"
            )
            log_proposals.append(log_density)

        log_proposal = torch.logsumexp(
            torch.stack(log_proposals, dim=1) + log_mixture_weights,
            dim=1,
        )
        return log_prior - log_proposal

    @staticmethod
    def _weights(
        log_prior_ratios: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        coefficient_mu: float,
        coefficient_sigma: float,
    ) -> np.ndarray:
        """Return importance weights with data-only counterweights."""

        log_counterweights = coefficient_mu * mu + coefficient_sigma * sigma
        return np.exp(log_prior_ratios - log_counterweights)

    @staticmethod
    def _effective_sample_fraction(weights: np.ndarray) -> float:
        """Return effective sample size divided by the number of weights."""

        return float(weights.sum() ** 2 / (np.square(weights).sum() * len(weights)))

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """Multiply SBI's ordinary per-example NLL by fitted sample weights."""

        if not isinstance(loss_args, LossArgsNPE):
            raise TypeError(
                "Expected type of loss_args to be LossArgsNPE,"
                f" but got {type(loss_args)}"
            )

        theta, x, masks, weights = (item.to(self._device) for item in batch)
        losses = self._loss(theta, x, masks, **asdict(loss_args))
        if loss_args.force_first_round_loss:
            return losses
        return weights * losses

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        """Return unweighted log probability; weights are stored in each batch."""

        theta = reshape_to_sample_batch_event(theta, theta.shape[1:])
        return self._neural_net.log_prob(theta, x).squeeze(dim=0)

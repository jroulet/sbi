"""Sequential neural posterior estimator with softened weights."""

import functools
from typing import Any, Callable, Optional, Union

import xgboost
import torch
from torch import Tensor
from torch.distributions import Distribution
import numpy as np

from scipy.optimize import differential_evolution

import sbi.utils as utils
from sbi.inference.posteriors import DirectPosterior
from sbi.inference.trainers.npe.npe_base import PosteriorEstimator
from sbi.inference.trainers.npe.npe_b import NPE_B
from sbi.neural_nets.estimators.shape_handling import reshape_to_sample_batch_event
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils.sbiutils import del_entries


class NPE_D(PosteriorEstimator):
    """Neural Posterior Estimation with softened weights."""

    @functools.wraps(NPE_B.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._a = None
        self._b = None
        self._log_proposal_to_prior_ratio_mu = xgboost.XGBRegressor()
        self._log_proposal_to_prior_ratio_sigma = xgboost.XGBRegressor()

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        """
        Return softened importance-weighted log probability.

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in
                order to train them with prior loss.
            proposal: Proposal distribution.

        Returns:
            Importance-weighted log-probability of the proposal posterior.
        """
        importance_weighted_log_prob = NPE_B._log_prob_proposal_posterior(
            self,
            theta=theta,
            x=x,
            masks=masks,
            proposal=proposal,
        )
        softening = self._get_softening(x)
        return softening * importance_weighted_log_prob

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ):
        """
        Like PosteriorEstimator.append_simulations, but also fits the
        log-proposal-to-prior-ratio regressor.
        """
        super().append_simulations(
            theta=theta,
            x=x,
            proposal=proposal,
            exclude_invalid_x=exclude_invalid_x,
            data_device=data_device,
        )

        self._fit_regressor()
        return self

    def _fit_regressor(self):
        theta = torch.concat(self._theta_roundwise, dim=0)
        x = torch.concat(self._x_roundwise, dim=0)

        self._round = max(self._data_round_index)
        log_proposal_to_prior_ratio = -self._get_log_importance_weights(
            theta).cpu().numpy()
        x = x.cpu().numpy()
        self._log_proposal_to_prior_ratio_mu.fit(
            x, log_proposal_to_prior_ratio)
        mu = self._log_proposal_to_prior_ratio_mu.predict(x)
        self._log_proposal_to_prior_ratio_sigma.fit(
            x, np.sqrt(np.abs(log_proposal_to_prior_ratio**2 - mu**2)))
        sigma = self._log_proposal_to_prior_ratio_sigma.predict(x)

        def _reweighting_efficiency(a, b, mu, sigma,
                                    log_proposal_to_prior_ratio):
            log_proposal_to_prior_ratio_pred = a * mu + b * sigma
            weights = np.exp(
                log_proposal_to_prior_ratio - log_proposal_to_prior_ratio_pred)
            n_effective = weights.sum()**2 / (weights**2).sum()
            efficiency = n_effective / len(weights)
            return efficiency

        result = differential_evolution(
            lambda y, *args: - _reweighting_efficiency(*y, *args),
            bounds=[(0, 2), (0, 4)],
            args=(mu, sigma, log_proposal_to_prior_ratio))
        a, b = result.x
        self._a = a
        self._b = b
        
        # # Print effective / total number of samples with & without regressor
        # weights_no_regressor = np.exp(
        #     log_proposal_to_prior_ratio - log_proposal_to_prior_ratio.max()
        # )
        # n_effective_no_regressor = (
        #     weights_no_regressor.sum()**2 / (weights_no_regressor**2).sum()
        # )
        # efficiency_no_regressor = n_effective_no_regressor / theta.size(0)

        # print(
        #     "Trained regressor for log proposal-to-prior ratio.\n"
        #     "Effective / total sample size: "
        #     f"{n_effective:.1f} / {theta.size(0)} "
        #     f"({efficiency:.2%})\n"
        #     "[Without the regressor it would have been: "
        #     f"{n_effective_no_regressor:.1f} / {theta.size(0)} "
        #     f"({efficiency_no_regressor:.2%})]"
        # )

    def _get_softening(self, x: Tensor) -> Tensor:
        """
        Predict proposal-to-prior ratio for x.

        Args:
            x: Batch of data.

        Returns:
            proposal-to-prior ratio.
        """
        x = x.cpu().numpy()

        mu = self._log_proposal_to_prior_ratio_mu.predict(x)
        sigma = self._log_proposal_to_prior_ratio_sigma.predict(x)

        log_softening = torch.tensor(self._a * mu + self._b * sigma,
                                     device=x.device)
        return torch.exp(log_softening)

    def _get_log_importance_weights(self, theta):
        """Return log(proposal/prior)."""
        # Heavily based on SNPE_B._log_prob_proposal_posterior (to the
        # point in which it could be refactored into a common function).

        # Evaluate prior
        # we accept prior log prob to be -Inf at theta
        # meaning that theta is out of the prior range (the weight is thus 0)
        log_prior = self._prior.log_prob(theta)
        utils.assert_not_nan_or_plus_inf(
            log_prior, "prior log probs of proposal samples"
        )

        # Evaluate proposal
        # (as theta comes from prior and proposal from previous rounds,
        # the last proposal is actually a mixture of the prior
        # and of all the previous proposals with coefficients representing
        # the proportion of the new theta added at each round)
        sizes = torch.tensor(
            [theta.size(0) for theta in self._theta_roundwise],
            device=theta.device,
        )
        proportions = sizes / sizes.sum()
        log_proportions = torch.log(proportions).repeat(theta.size(0), 1)

        log_previous_proposals = torch.zeros(
            (theta.size(0), self._round + 1), device=theta.device
        )
        for k, density in enumerate(self._proposal_roundwise):
            # we accept the k th proposal log prob to be -Inf at theta
            # meaning that theta is out of the k th proposal range
            log_previous_proposals[:, k] = density.log_prob(theta)
            utils.assert_not_nan_or_plus_inf(
                log_previous_proposals[:, k],
                "proposal log probs of proposal samples"
            )
        log_proposal = torch.logsumexp(
            log_proportions + log_previous_proposals, dim=1
        )

        return log_prior - log_proposal

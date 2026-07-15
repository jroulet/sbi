import numpy as np
import torch
from torch.distributions import MultivariateNormal

from sbi.inference import NPE_D
from sbi.inference.trainers.npe import NPE_D as TrainerNPE_D


def test_npe_d_is_publicly_exported():
    assert NPE_D is TrainerNPE_D


def test_npe_d_labrador_weight_formula():
    log_ratios = np.array([-2.0, 0.0, 2.0])
    mu = np.array([-1.0, 0.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])

    weights = NPE_D._weights(log_ratios, mu, sigma, 1.0, 2.0)

    np.testing.assert_allclose(weights, np.exp(log_ratios - mu - 2.0 * sigma))


def test_npe_d_fits_sequential_counterweights():
    torch.manual_seed(0)
    prior = MultivariateNormal(torch.zeros(1), torch.eye(1))
    proposal = MultivariateNormal(torch.ones(1), 0.25 * torch.eye(1))
    inference = NPE_D(
        prior=prior,
        xgb_regressor_kwargs={
            "n_estimators": 4,
            "max_depth": 2,
            "n_jobs": 1,
            "random_state": 0,
        },
        show_progress_bars=False,
    )

    theta_prior = prior.sample((100,))
    theta_proposal = proposal.sample((100,))
    inference.append_simulations(
        theta_prior, theta_prior + 0.1 * torch.randn_like(theta_prior), proposal=prior
    )
    inference.append_simulations(
        theta_proposal,
        theta_proposal + 0.1 * torch.randn_like(theta_proposal),
        proposal=proposal,
    )
    inference._round = 1

    train_loader, validation_loader = inference.get_dataloaders(
        training_batch_size=20,
        validation_fraction=0.2,
    )

    assert len(next(iter(train_loader))) == 4
    assert len(next(iter(validation_loader))) == 4
    assert torch.isfinite(inference._counterweight_weights).all()
    assert inference.counterweight_efficiency >= inference.raw_importance_efficiency

    estimator = inference.train(
        training_batch_size=20,
        validation_fraction=0.2,
        stop_after_epochs=1,
        max_num_epochs=1,
    )
    assert estimator is inference._neural_net

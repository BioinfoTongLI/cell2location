from typing import Optional

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from scipy.sparse import csr_matrix
from scvi import REGISTRY_KEYS
from scvi.nn import one_hot

# class NegativeBinomial(TorchDistributionMixin, ScVINegativeBinomial):
#    pass


class LocationModelWTAMultiExperimentHierarchicalGeneLevel(PyroModule):
    r"""
    Cell2location models the elements of :math:`D` as Negative Binomial distributed,
    given an unobserved gene expression level (rate) :math:`mu` and a gene- and batch-specific
    over-dispersion parameter :math:`\alpha_{e,g}` which accounts for unexplained variance:

    .. math::
        D_{s,g} \sim \mathtt{NB}(\mu_{s,g}, \alpha_{e,g})

    The expression level of genes :math:`\mu_{s,g}` in the mRNA count space is modelled
    as a linear function of expression signatures of reference cell types :math:`g_{f,g}`:

    .. math::
        \mu_{s,g} = (m_{g} \left (\sum_{f} {w_{s,f} \: g_{f,g}} \right) + s_{e,g}) y_{s}

    Here, :math:`w_{s,f}` denotes regression weight of each reference signature :math:`f` at location :math:`s`, which can be interpreted as the expected number of cells at location :math:`s` that express reference signature :math:`f`;
    :math:`g_{f,g}` denotes the reference signatures of cell types :math:`f` of each gene :math:`g`, `cell_state_df` input ;
    :math:`m_{g}` denotes a gene-specific scaling parameter which adjusts for global differences in sensitivity between technologies (platform effect);
    :math:`y_{s}` denotes a location/observation-specific scaling parameter which adjusts for differences in sensitivity between observations and batches;
    :math:`s_{e,g}` is additive component that account for gene- and location-specific shift, such as due to contaminating or free-floating RNA.

    To account for the similarity of location patterns across cell types, :math:`w_{s,f}` is modelled using
    another layer  of decomposition (factorization) using :math:`r={1, .., R}` groups of cell types,
    that can be interpreted as cellular compartments or tissue zones. Unless stated otherwise, R is set to 50.

    Corresponding graphical model can be found in supplementary methods:
    https://www.biorxiv.org/content/10.1101/2020.11.15.378125v1.supplementary-material

    Approximate Variational Inference is used to estimate the posterior distribution of all model parameters.

    Estimation of absolute cell abundance :math:`w_{s,f}` is guided using informed prior on the number of cells
    (argument called `N_cells_per_location`). It is a tissue-level global estimate, which can be derived from histology
    images (H&E or DAPI), ideally paired to the spatial expression data or at least representing the same tissue type.
    This parameter can be estimated by manually counting nuclei in a 10-20 locations in the histology image
    (e.g. using 10X Loupe browser), and computing the average cell abundance.
    An appropriate setting of this prior is essential to inform the estimation of absolute cell type abundance values,
    however, the model is robust to a range of similar values.
    In settings where suitable histology images are not available, the size of capture regions relative to
    the expected size of cells can be used to estimate `N_cells_per_location`.

    The prior on detection efficiency per location :math:`y_s` is selected to discourage over-normalisation, such that
    unless data has evidence of strong technical effect, the effect is assumed to be small and close to
    the mean sensitivity for each batch :math:`y_e`:

    .. math::
        y_s \sim Gamma(detection\_alpha, detection\_alpha / y_e)

    where y_e is unknown/latent average detection efficiency in each batch/experiment:

    .. math::
        y_e \sim Gamma(10, 10 / detection\_mean)

    """

    # training mode without observed data (just using priors)
    training_wo_observed = False
    training_wo_initial = False

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        n_neg_probes,
        cell_state_mat,
        n_groups: int = 50,
        detection_mean=1 / 2,
        detection_alpha=20.0,
        m_g_gene_level_prior={"mean": 1, "mean_var_ratio": 1.0, "alpha_mean": 3.0, "slide_alpha": 20.0},
        A_factors_per_location=7.0,
        B_groups_per_location=7.0,
        N_cells_mean_var_ratio=0.25,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 10.0},
        w_sf_mean_var_ratio=5.0,
        init_vals: Optional[dict] = None,
        init_alpha=20.0,
        dropout_p=0.0,
    ):

        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_groups = n_groups
        self.n_neg_probes = n_neg_probes

        self.m_g_gene_level_prior = m_g_gene_level_prior

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.w_sf_mean_var_ratio = w_sf_mean_var_ratio
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        detection_hyp_prior["mean"] = detection_mean
        detection_hyp_prior["alpha"] = detection_alpha
        self.detection_hyp_prior = detection_hyp_prior

        self.dropout_p = dropout_p
        if self.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=self.dropout_p)

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
            self.init_alpha = init_alpha
            self.register_buffer("init_alpha_tt", torch.tensor(self.init_alpha))

        factors_per_groups = A_factors_per_location / B_groups_per_location

        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_alpha"] / self.detection_hyp_prior["mean"]),
        )

        # compute hyperparameters from mean and sd
        self.register_buffer("m_g_mu_hyp", torch.tensor(self.m_g_gene_level_prior["mean"]))
        self.register_buffer("slide_alpha", torch.tensor(self.m_g_gene_level_prior["slide_alpha"]))
        self.register_buffer(
            "m_g_mu_mean_var_ratio_hyp",
            torch.tensor(self.m_g_gene_level_prior["mean_var_ratio"]),
        )

        self.register_buffer("m_g_alpha_hyp_mean", torch.tensor(self.m_g_gene_level_prior["alpha_mean"]))

        self.cell_state_mat = cell_state_mat
        self.register_buffer("cell_state", torch.tensor(cell_state_mat.T))

        self.register_buffer("factors_per_groups", torch.tensor(factors_per_groups))
        self.register_buffer("B_groups_per_location", torch.tensor(B_groups_per_location))
        self.register_buffer("N_cells_mean_var_ratio", torch.tensor(N_cells_mean_var_ratio))

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        self.register_buffer("w_sf_mean_var_ratio_tensor", torch.tensor(self.w_sf_mean_var_ratio))

        self.register_buffer("n_factors_tensor", torch.tensor(self.n_factors))
        self.register_buffer("n_groups_tensor", torch.tensor(self.n_groups))
        
        self.register_buffer("neg_probe_alpha", 10**6*torch.ones((self.n_obs, self.n_neg_probes)))
        self.register_buffer("one", torch.tensor(1.))
        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones_1_n_groups", torch.ones((1, self.n_groups)))
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        neg_data = tensor_dict['neg_probes']
        n_nuclei = tensor_dict['nuclei']
        return (x_data, neg_data, n_nuclei, ind_x, batch_index), {}

    def create_plates(self, x_data, neg_data, n_nuclei, idx, batch_index):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """
        Create a dictionary with:

        1. "name" - the name of observation/minibatch plate;
        2. "input" - indexes of model args to provide to encoder network when using amortised inference;
        3. "sites" - dictionary with

          * keys - names of variables that belong to the observation plate (used to recognise
            and merge posterior samples for minibatch variables)
          * values - the dimensions in non-plate axis of each variable (used to construct output
            layer of encoder network when using amortised inference)
        """

        return {
            "name": "obs_plate",
            "input": [0, 2],  # expression data + (optional) batch index
            "input_transform": [
                torch.log1p,
                lambda x: x,
            ],  # how to transform input data before passing to NN
            "input_normalisation": [
                False,
                False,
            ],  # whether to normalise input data before passing to NN
            "sites": {
                "n_s_cells_per_location": 1,
                "b_s_groups_per_location": 1,
                "z_sr_groups_factors": self.n_groups,
                "w_sf": self.n_factors,
                "detection_y_s": 1,
            },
        }

    def forward(self, x_data, neg_data, n_nuclei, idx, batch_index):

        obs2sample = one_hot(batch_index, self.n_batch)

        obs_plate = self.create_plates(x_data, neg_data, n_nuclei, idx, batch_index)
        
        # ============================ Negative Probe Binding ===================== #
        # Negative probe counts scale linearly with the total number of counts in a region of interest.
        # The linear slope is drawn from a gamma distribution. Mean and variance are inferred from the data
        # and are the same for the non-specific binding term for gene probes further below.
        total_counts_r = torch.sum(x_data, axis = 1).unsqueeze(-1)
        b_n_hyper_1 = pyro.sample('b_n_hyper_1', dist.Gamma(self.one/0.33, self.one))
        b_n_hyper_2 = pyro.sample('b_n_hyper_2', dist.Gamma(self.one, self.one))
        b_n = pyro.sample('b_n', dist.Gamma((b_n_hyper_1/b_n_hyper_2)**2,
                                             b_n_hyper_1/b_n_hyper_2**2 
                                           ).expand([self.n_batch, self.n_neg_probes]).to_event(2))
        y_rn = torch.einsum('ij, jk -> ik', obs2sample, b_n) * total_counts_r

        # =====================Gene expression level scaling m_g======================= #
        # Explains difference in sensitivity for each gene between single cell and spatial technology
        m_g_mean = pyro.sample(
            "m_g_mean",
            dist.Gamma(
                self.m_g_mu_mean_var_ratio_hyp * self.m_g_mu_hyp,
                self.m_g_mu_mean_var_ratio_hyp,
            )
            .expand([1, 1])
            .to_event(2),
        )  # (1, 1)

        m_g_alpha_e_inv = pyro.sample(
            "m_g_alpha_e_inv",
            dist.Exponential(self.m_g_alpha_hyp_mean).expand([1, 1]).to_event(2),
        )  # (1, 1)
        m_g_alpha_e = self.ones / m_g_alpha_e_inv.pow(2)
        
        # global effect on each gene:
        m_g = pyro.sample(
            "m_g",
            dist.Gamma(m_g_alpha_e, m_g_alpha_e / m_g_mean).expand([1, self.n_vars]).to_event(2),  # self.m_g_mu_hyp)
        )  # (1, n_vars)

        # independent experiment-specific effect on each gene (narrow prior around 1)
        m_ge = pyro.sample('m_ge', dist.Gamma(self.one*100, self.one*100).expand([self.n_batch, self.n_vars]).to_event(2))
        # experiment specific capture efficiency (wide prior around 1)
        m_e = pyro.sample('m_e', dist.Gamma(self.slide_alpha, self.slide_alpha).expand([self.n_batch, 1]).to_event(2))

        # =====================Cell abundances w_sf======================= #
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        with obs_plate as ind:
            k = "n_s_cells_per_location"
            n_s_cells_per_location = pyro.sample(
                k,
                dist.Gamma(
                    n_nuclei * self.N_cells_mean_var_ratio,
                    self.N_cells_mean_var_ratio,
                ),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=n_s_cells_per_location,
                )  # (self.n_obs, self.n_groups)

            k = "b_s_groups_per_location"
            b_s_groups_per_location = pyro.sample(
                k,
                dist.Gamma(self.B_groups_per_location, self.ones),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=b_s_groups_per_location,
                )  # (self.n_obs, self.n_groups)

        # cell group loadings
        shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
        rate = self.ones_1_n_groups / (n_s_cells_per_location / b_s_groups_per_location)
        with obs_plate as ind:
            k = "z_sr_groups_factors"
            z_sr_groups_factors = pyro.sample(
                k,
                dist.Gamma(shape, rate),  # .to_event(1)#.expand([self.n_groups]).to_event(1)
            )  # (n_obs, n_groups)

            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=z_sr_groups_factors,
                )  # (self.n_obs, self.n_groups)

        k_r_factors_per_groups = pyro.sample(
            "k_r_factors_per_groups",
            dist.Gamma(self.factors_per_groups, self.ones).expand([self.n_groups, 1]).to_event(2),
        )  # (self.n_groups, 1)

        c2f_shape = k_r_factors_per_groups / self.n_factors_tensor

        x_fr_group2fact = pyro.sample(
            "x_fr_group2fact",
            dist.Gamma(c2f_shape, k_r_factors_per_groups).expand([self.n_groups, self.n_factors]).to_event(2),
        )  # (self.n_groups, self.n_factors)

        with obs_plate as ind:
            w_sf_mu = z_sr_groups_factors @ x_fr_group2fact

            k = "w_sf"
            w_sf = pyro.sample(
                k,
                dist.Gamma(
                    w_sf_mu * self.w_sf_mean_var_ratio_tensor,
                    self.w_sf_mean_var_ratio_tensor,
                ),
            )  # (self.n_obs, self.n_factors)
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=w_sf,
                )  # (self.n_obs, self.n_factors)

        # =====================Location-specific detection efficiency ======================= #
        # y_s with hierarchical mean prior
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Gamma(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        detection_hyp_prior_alpha = pyro.deterministic(
            "detection_hyp_prior_alpha",
            self.ones_n_batch_1 * self.detection_hyp_prior_alpha,
        )

        beta = (obs2sample @ detection_hyp_prior_alpha) / (obs2sample @ detection_mean_y_e)
        with obs_plate:
            k = "detection_y_s"
            detection_y_s = pyro.sample(
                k,
                dist.Gamma(obs2sample @ detection_hyp_prior_alpha, beta),
            )  # (self.n_obs, 1)

            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                # pre-training Variational distribution to initial values
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=detection_y_s,
                )  # (self.n_obs, 1)

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by
        # cell state signatures (e.g. background, free-floating RNA)
        s_g_gene_add = pyro.sample('s_g_gene_add', dist.Gamma((b_n_hyper_1/b_n_hyper_2)**2,
                                     b_n_hyper_1/b_n_hyper_2**2 
                                   ).expand([self.n_batch, self.n_vars]).to_event(2))

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([self.n_batch, self.n_vars]).to_event(2),
        )  # (self.n_batch, self.n_vars)

        # =====================Expected expression ======================= #
        if not self.training_wo_observed:
            # expected expression
            mu_biol = ((w_sf @ self.cell_state) * m_g \
                       * (obs2sample @ m_e) * (obs2sample @ m_ge) \
                       + (obs2sample @ s_g_gene_add)*total_counts_r) * detection_y_s
            mu = torch.concat([y_rn, mu_biol], axis = 1)
            alpha_biol = obs2sample @ (self.ones / alpha_g_inverse.pow(2))
            alpha = torch.concat([self.neg_probe_alpha, alpha_biol], axis = 1)

            # =====================DATA likelihood ======================= #
            # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
            if self.dropout_p != 0:
                x_data = self.dropout(x_data)
            with obs_plate:
                pyro.sample(
                    "data_target",
                    dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                    obs=torch.concat([neg_data, x_data], axis = 1))
        
        # =====================Compute mRNA count from each factor in locations  ======================= #
        with obs_plate:
            mRNA = w_sf * (self.cell_state * m_g).sum(-1)
            pyro.deterministic("u_sf_mRNA_factors", mRNA)

    def compute_expected(self, samples, adata_manager, ind_x=None):
        r"""Compute expected expression of each gene in each location. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        mu = (
            np.dot(samples["w_sf"][ind_x, :], self.cell_state_mat.T) * samples["m_g"]
            + np.dot(obs2sample, samples["s_g_gene_add"])
        ) * samples["detection_y_s"][ind_x, :]
        alpha = np.dot(obs2sample, 1 / np.power(samples["alpha_g_inverse"], 2))

        return {"mu": mu, "alpha": alpha, "ind_x": ind_x}

    def compute_expected_per_cell_type(self, samples, adata_manager, ind_x=None):
        r"""
        Compute expected expression of each gene in each location for each cell type.

        Parameters
        ----------
        samples
            Posterior distribution summary self.samples[f"post_sample_q05}"]
            (or 'means', 'stds', 'q05', 'q95') produced by export_posterior().
        ind_x
            Location/observation indices for which to compute expected count
            (if None all locations are used).

        Returns
        -------
        dict
          dictionary with:

            1. list with expected expression counts (sparse, shape=(N locations, N genes)
               for each cell type in the same order as mod\.factor_names_;
            2. np.array with location indices
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)

        # fetch data
        x_data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        x_data = csr_matrix(x_data)

        # compute total expected expression
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        mu = np.dot(samples["w_sf"][ind_x, :], self.cell_state_mat.T) * samples["m_g"] + np.dot(
            obs2sample, samples["s_g_gene_add"]
        )

        # compute conditional expected expression per cell type
        mu_ct = [
            csr_matrix(
                x_data.multiply(
                    (
                        np.dot(
                            samples["w_sf"][ind_x, i, np.newaxis],
                            self.cell_state_mat.T[np.newaxis, i, :],
                        )
                        * samples["m_g"]
                    )
                    / mu
                )
            )
            for i in range(self.n_factors)
        ]

        return {"mu": mu_ct, "ind_x": ind_x}

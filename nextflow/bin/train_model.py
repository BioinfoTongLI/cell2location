#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Tong LI <tongli.bioinfo@protonmail.com>
#
# Distributed under terms of the BSD-3 license.

"""
Get cell type signature from reference h5ad
"""
import fire
import sys

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import cell2location
# import scvi
# print(scvi.__version__)

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text

from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel


def convert_to_int_count(adata_ref):
    # get counts from raw slot
    adata_snrna_raw_new = sc.AnnData(adata_ref.X)
    adata_snrna_raw_new.obs = adata_ref.obs
    adata_snrna_raw_new.var = adata_ref.var
    adata_snrna_raw_new.obsm = adata_ref.obsm
    adata_snrna_raw_new.uns = adata_ref.uns
    adata_snrna_raw_new.obsp = adata_ref.obsp
    adata_snrna_raw_new.varm = adata_ref.varm

    adata_snrna_raw = adata_snrna_raw_new.copy()

    # revert log-transform
    adata_snrna_raw.X.data = np.expm1(adata_snrna_raw.X.data)

    # revert normalisation
    adata_snrna_raw.X = (adata_snrna_raw.X / 10000).multiply(adata_snrna_raw.obs['n_counts'].values.reshape((adata_snrna_raw.n_obs,1)))

    # checking that data is integer counts
    print(adata_snrna_raw.X.data)

    # convert to integer counts
    adata_snrna_raw.X.data = np.round(adata_snrna_raw.X.data)

    from scipy.sparse import csr_matrix
    adata_snrna_raw.X = csr_matrix(adata_snrna_raw.X)
    adata_snrna_raw.X.data = adata_snrna_raw.X.data.astype(int)
    return adata_snrna_raw


def main(stem, h5ad_ref,
        is_raw_count=False,
        cell_count_cutoff=1,
        cell_percentage_cutoff2=0.01):
    adata_ref = sc.read(h5ad_ref)
    if not is_raw_count:
        adata_ref = convert_to_int_count(adata_ref)

    # before we estimate the reference cell type signature we recommend to perform very permissive genes selection
    # in this 2D histogram orange rectangle lays over excluded genes.
    # In this case, the downloaded dataset was already filtered using this method,
    # hence no density under the orange rectangle
    selected = filter_genes(adata_ref, cell_count_cutoff=cell_count_cutoff, cell_percentage_cutoff2=cell_percentage_cutoff2, nonz_mean_cutoff=1.12)

    # filter the object
    adata_ref = adata_ref[:, selected].copy()

    # prepare anndata for the regression model
    # scvi.data.setup_anndata(adata=adata_ref,
                            # # 10X reaction / sample / batch
                            # batch_key='adj_sample',
                            # # cell type, covariate used for constructing signatures
                            # labels_key='leiden_R_anno_id',
                            # # multiplicative technical effects (platform, 3' vs 5', donor effect)
                            # categorical_covariate_keys=['batch']
                           # )
    # scvi.data.view_anndata_setup(adata_ref)

    # create and train the regression model
    RegressionModel.setup_anndata(adata_ref)
    mod = RegressionModel(adata_ref)

    # Use all data for training (validation not implemented yet, train_size=1)
    mod.train(max_epochs=300, batch_size=2500, train_size=1, lr=0.002, use_gpu=True)

    # plot ELBO loss history during training, removing first 20 epochs from the plot
    # mod.plot_history(20)

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_ref = mod.export_posterior(
        adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
    )

    # Save model
    mod.save(f"{stem}_ref_model", overwrite=True)

    # Save anndata object with results
    adata_ref.write(f"{stem}_sc_signature.h5ad")


if __name__ == "__main__":
    fire.Fire(main)

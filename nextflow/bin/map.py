#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Tong LI <tongli.bioinfo@protonmail.com>
#
# Distributed under terms of the BSD-3 license.

"""

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
import scvi

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
import seaborn as sns


def main(stem, signature, iss_count):
    adata_ref = sc.read_h5ad(signature)
    adata_iss = sc.read_h5ad(iss_count)
    # print(adata_ref)

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_iss.var_names, inf_aver.index)
    adata_iss = adata_iss[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
    print(inf_aver, adata_iss)

    # create and train the model
    cell2location.models.Cell2location.setup_anndata(adata_iss)
    mod = cell2location.models.Cell2location(
        adata_iss, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=30,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection (using default here):
        detection_alpha=200
    )

    mod.train(max_epochs=30000,
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
              use_gpu=True)

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_iss = mod.export_posterior(
        adata_iss, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
    )

    # Save model
    mod.save(f"{stem}", overwrite=True)
    adata_iss.write(f"{stem}_mapped_cell_types.h5ad")


if __name__ == "__main__":
    fire.Fire(main)

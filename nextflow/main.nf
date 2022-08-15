#!/usr/bin/env/ nextflow
// Copyright © 2022 Tong LI <tongli.bioinfo@protonmail.com>

nextflow.enable.dsl=2

params.ref_h5ad = "/nfs/team283_imaging/HZ_HLB/playground_Tong/HZ_HLB_hindlimb_20220130_63x_fine_tune/C2L_signature/spatial_mapping_alphatest20/sp.h5ad"
params.ISS_counts = "/nfs/team283_imaging/HZ_HLB/playground_Tong/HZ_HLB_hindlimb_20220304_63x_label_image_countTable/"

// All the training parameters that will be crossed for training
params.cell_count_cutoffs = [1, 2, 3]
params.cell_percentage_cutoff2 = [0.01, 0.02, 0.03]

process train {
    echo true

    container "${projectDir}/c2l.sif"
    containerOptions "--nv"
    /*publishDir params.outdir, mode:"copy"*/
    storeDir params.outdir

    input:
    path(h5ad_ref)
    each cell_count_cutoff
    each cell_percentage_cutoff2

    output:
    tuple val(stem), path("${stem}_sc_signature.h5ad"), emit: signature
    path("${stem}_ref_model")

    script:
    stem = h5ad_ref.baseName
    """
    train_model.py --stem ${stem} --h5ad_ref ${h5ad_ref} \
        --cell_count_cutoff ${cell_count_cutoff} \
        --cell_percentage_cutoff2 ${cell_percentage_cutoff2}
    """
}


process map_cell_types {
    echo true

    container "${projectDir}/c2l.sif"
    containerOptions "--nv"
    publishDir params.outdir, mode:"copy"

    input:
    tuple val(stem), path(signature)
    path(raw_ISS_count)

    output:
    tuple val(stem), path("${stem}_mapped_cell_types.h5ad")

    script:
    """
    map.py --stem ${stem} --signature ${signature} --iss_count ${raw_ISS_count}
    """
}

workflow {
    train(channel.fromPath(params.ref_h5ad),
        channel.from(params.cell_count_cutoffs),
        channel.from(cell_percentage_cutoff2))
    map_cell_types(train.out.signature, channel.fromPath(params.ISS_counts))
}

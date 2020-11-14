import os.path as op


def get_tractogram_filename(myafq, bundle_name):
    row = myafq.data_frame.iloc[0]

    bundles_dir = op.join(row['results_dir'], 'bundles')

    fname = op.split(
        myafq._get_fname(
            row,
            f'-{bundle_name}'
            f'_tractography.trk',
            include_track=True,
            include_seg=True
        )
    )

    tg_fname = op.join(bundles_dir, fname[1])

    return tg_fname


def get_scalar_filename(myafq, scalar):
    row = myafq.data_frame.iloc[0]

    scalar_fname = myafq._get_fname(
        row,
        f'_model-{scalar}.nii.gz'
    )

    return scalar_fname

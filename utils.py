import os
import os.path as op

import re
from glob import glob
from fnmatch import fnmatch

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

from AFQ import api
import AFQ.data as afd

dataset_names = [
    'HARDI',
    'HCP',
    'HCP_retest'
]

dataset_homes = {
    'HARDI': 'stanford_hardi',
    'HCP': 'HCP_1200',
    'HCP_retest': 'hcp_retest'
}

dataset_subjects = {
    'HARDI': ['01'],
    'HCP': [
        '103818', '105923', '111312', '114823', '115320',
        '122317', '125525', '130518', '135528', '137128',
        '139839', '143325', '144226', '146129', '149337',
        '149741', '151526', '158035', '169343', '172332',
        '175439', '177746', '185442', '187547', '192439',
        '194140', '195041', '200109', '200614', '204521',
        '250427', '287248', '341834', '433839', '562345',
        '599671', '601127', '627549', '660951', '662551',
        '783462', '859671', '861456', '877168', '917255'
    ],
    'HCP_retest': [
        '103818', '105923', '111312', '114823', '115320',
        '122317', '125525', '130518', '135528', '137128',
        '139839', '143325', '144226', '146129', '149337',
        '149741', '151526', '158035', '169343', '172332',
        '175439', '177746', '185442', '187547', '192439',
        '194140', '195041', '200109', '200614', '204521',
        '250427', '287248', '341834', '433839', '562345',
        '599671', '601127', '627549', '660951', '662551',
        '783462', '859671', '861456', '877168', '917255'
    ]
}

dataset_subjects_medium = {
    'HARDI': ['01'],
    'HCP': ['103818', '105923', '111312', '114823', '115320'],
    'HCP_retest': ['103818', '105923', '111312', '114823', '115320']
}

dataset_subjects_small = {
    'HARDI': ['01'],
    'HCP': ['103818', '105923'],
    'HCP_retest': ['103818', '105923']
}

derivative_names = ['afq', 'dmriprep']

bundle_names = ['SLF_L', 'SLF_R']


def get_dataset_home(dataset_name):
    return op.join(afd.afq_home, dataset_homes[dataset_name])


def get_dmriprep_home(dataset_name):
    if dataset_name == 'HARDI':
        dataset_home = 'vistasoft'
    else:
        dataset_home = 'dmriprep'

    return op.join(get_dataset_home(dataset_name), 'derivatives', dataset_home)


def get_afq_home(dataset_name):
    return op.join(get_dataset_home(dataset_name), 'derivatives', 'afq')


def get_subject_session_home(derivatives_home, subject, session='ses-01'):
    return op.join(derivatives_home, subject, session)


def get_subject_session_dwi_file(dataset_name, subject_session_home, subject, session='ses-01'):
    if dataset_name == 'HARDI':
        dwi_file = f'{subject}_{session}_dwi.nii.gz'
    else:
        dwi_file = f'{subject}_dwi.nii.gz'

    return op.join(subject_session_home, 'dwi', dwi_file)


def get_afq(dataset_name):
    if dataset_name == 'HARDI':
        return api.AFQ(
            bids_path=op.join(afd.afq_home, 'stanford_hardi'),
            dmriprep='vistasoft'
        )
    elif dataset_name == 'HCP':
        return api.AFQ(
            bids_path=op.join(afd.afq_home, 'HCP_1200'),
            dmriprep='dmriprep'
        )
    elif dataset_name == 'HCP_retest':
        return api.AFQ(
            bids_path=op.join(afd.afq_home, 'hcp_retest'),
            dmriprep='dmriprep'
        )

    raise Exception(f'{dataset_name} not supported')


def display_dwi_slice(dataset_name, dwi_nii):
    dwi_image_data = dwi_nii.get_fdata()

    if len(dwi_image_data.shape) == 4:
        dwi_image_data = dwi_image_data[..., int(dwi_image_data.shape[3] / 2)]

    n_i, n_j, n_k = dwi_image_data.shape
    center_k = (n_k - 1) // 2
    slice = dwi_image_data[:, :, center_k]

    plt.figure()
    plt.title(dataset_name)
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.show()


def get_subjects(dataset_name):
    return dataset_subjects[dataset_name]


def get_subjects_medium(dataset_name):
    return dataset_subjects_medium[dataset_name]


def get_subjects_small(dataset_name):
    return dataset_subjects_small[dataset_name]


def fetch_data(dataset_name, subjects):
    if dataset_name == 'HARDI':
        afd.organize_stanford_data()
    elif dataset_name == 'HCP':
        afd.fetch_hcp(subjects)
    elif dataset_name == 'HCP_retest':
        afd.fetch_hcp(subjects, study='HCP_Retest')

    raise Exception(f'{dataset_name} not supported')


def get_hcp_s3_url(dataset_name, subject):
    if dataset_name == 'HCP':
        return f's3://profile-hcp-west/hcp_reliability/single_shell/hcp_1200_afq/sub-{subject}/'
    elif dataset_name == 'HCP_retest':
        return f's3://profile-hcp-west/hcp_reliability/single_shell/hcp_retest_afq/sub-{subject}/'

    raise Exception(f'{dataset_name} not supported')


def get_iloc(myafq, subject):
    iloc, = myafq.data_frame.index[myafq.data_frame['subject'] == subject]
    return iloc


def make_dirs(myafq, dataset_name, bundle_name, subjects=None):
    if subjects is None:
        subjects = myafq.subjects

    for sub in subjects:
        for ses in myafq.sessions:
            target_dir = op.join(
                'subbundles',
                dataset_name,
                bundle_name,
                sub,
                ses
            )
            os.makedirs(target_dir, exist_ok=True)


def get_dir_name(myafq, dataset_name, bundle_name, loc):
    target_dir = op.join(
        'subbundles',
        dataset_name,
        bundle_name,
        myafq.data_frame.at[loc, 'subject'],
        myafq.data_frame.at[loc, 'ses']
    )

    if not op.exists(target_dir):
        raise Exception(f'{target_dir} does not exist')

    return target_dir


def get_tractogram_filename(myafq, bundle_name, loc):
    if not myafq.bundle_dict.get(bundle_name):
        raise Exception(f'{bundle_name} not found')

    row = myafq.data_frame.iloc[loc]

    # Use clean bundles
    bundles_dir = op.join(row['results_dir'], 'clean_bundles')

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

    if not op.exists(tg_fname):
        raise Exception(f'{tg_fname} does not exist')

    return tg_fname


def get_scalar_filename(myafq, scalar, loc):
    row = myafq.data_frame.iloc[loc]

    if scalar not in myafq.scalars:
        raise Exception(f'{scalar} not supported')

    scalar_fname = myafq._get_fname(
        row,
        f'_model-{scalar}.nii.gz'
    )

    if not op.exists(scalar_fname):
        raise Exception(f'{scalar_fname} does not exist')

    return scalar_fname


def add_colorbar(im, aspect=5, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def streamline_tract_profile_test(values):
    print(values.shape)
    print('min: ', np.min(values))
    print('max: ', np.max(values))

    plt.figure()
    plt.imshow(values.T, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title('$\\mu$')
    plt.plot(np.mean(values, 0))
    plt.show()


def adjacency_test(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = adjacency.to_numpy()

    plt.figure()
    plt.hist(adjacency.ravel())
    plt.show()

    print('symmetric: ', not(np.sum(abs(adjacency-adjacency.T))))
    print('finite: ', np.isfinite(adjacency).any())
    print('min: ', np.min(adjacency))
    print('max: ', np.max(adjacency))


def get_adjacencies_names(target_dir, pattern=None):
    adjacencies_names = []

    p = re.compile('adjacency_(.*).npy')

    for adjancency_filename in natural_sort(glob(op.join(target_dir, 'adjacency*.npy'))):
        if pattern is not None and not fnmatch(adjancency_filename, pattern):
            continue

        m = p.search(adjancency_filename)
        adjacencies_names.append(m.group(1))

    return adjacencies_names


def get_adjacencies(target_dir, pattern=None):
    adjacencies = []

    for adjancency_filename in natural_sort(glob(op.join(target_dir, 'adjacency*.npy'))):
        if pattern is not None and not fnmatch(adjancency_filename, pattern):
            continue
        adjacencies.append(np.load(adjancency_filename))

    return adjacencies


def natural_sort(list_to_sort):
    # def convert(text): int(text) if text.isdigit() else text.lower()
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # def alphanum_key(key): [convert(c) for c in re.split('([0-9]+)', key)]
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_to_sort, key=alphanum_key)


def resort_cluster_ids(idx):
    from_values = np.flip(np.argsort(np.bincount(idx))[-(np.unique(idx).size):])
    to_values = np.arange(from_values.size)
    d = dict(zip(from_values, to_values))
    new_idx = np.copy(idx)
    for k, v in d.items(): new_idx[idx==k] = v
    return new_idx
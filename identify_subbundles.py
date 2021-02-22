

###############################################################################
###############################################################################
# Now that have clusters for each subject, from one or more expirements/models:
#
# Data driven approach to identify subbundles across subjects 
#    from the HCP test clusters
# • Load each subjects tractography file and load cluster id, or alteriantively,
#    Load all cluster tractography files for each subject
# • Move each cluster into MNI space
# • Use weighted dice coefficient --
#    to calculate the pairwise overlap across subjects for each cluster
###############################################################################
###############################################################################

def move_tractogram_to_MNI_space(data_dir, subject, tractogram):
    from os.path import join, expanduser
    import numpy as np
    import nibabel as nib
    import AFQ.data as afd
    import AFQ.registration as reg
    import dipy.tracking.streamline as dts
    import dipy.tracking.utils as dtu
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.io.stateful_tractogram import Space

    ###############################################################################
    afq_base_dir = join(expanduser('~'),'AFQ_data')
    afq_derivatives_base_dir = join(afq_base_dir, data_dir, 'derivatives')

    ###############################################################################
    subject_path = join(f'sub-{subject}', 'ses-01')

    ###############################################################################
    # load subject dwi image
    dwi_derivatives_dir = join(afq_derivatives_base_dir, 'dmriprep')
    dwi_file = join(dwi_derivatives_dir, subject_path, 'dwi', f'sub-{subject}_dwi.nii.gz')

    # ~/AFQ_data/HCP_1200/derivatives/dmriprep/sub-125525/ses-01/dwi/sub-125525_dwi.nii.gz
    dwi_img = nib.load(dwi_file)

    ## validate dwi image
    # dwi_img.shape -- (145, 174, 145, 288)
    # dwi_img.affine
    # array([[  -1.25,    0.  ,    0.  ,   90.  ],
    #        [   0.  ,    1.25,    0.  , -126.  ],
    #        [   0.  ,    0.  ,    1.25,  -72.  ],
    #        [   0.  ,    0.  ,    0.  ,    1.  ]])

    ###############################################################################
    # load subject to MNI mapping from AFQ

    # ~/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz
    MNI_T2_img = afd.read_mni_template()

    ## validate MNI image
    # MNI_T2_img.shape --  (193, 229, 193)
    # MNI_T2_img.affine
    # array([[   1.,    0.,    0.,  -96.],
    #        [   0.,    1.,    0., -132.],
    #        [   0.,    0.,    1.,  -78.],
    #        [   0.,    0.,    0.,    1.]])

    afq_derivatives_dir = join(afq_derivatives_base_dir, 'afq')

    # ~/AFQ_data/HCP_1200/derivatives/afq/sub-125525/ses-01/sub-125525_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz
    mapping_file = join(afq_derivatives_dir, subject_path, f'sub-{subject}_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz')

    # ~/AFQ_data/HCP_1200/derivatives/afq/sub-125525/ses-01/sub-125525_dwi_prealign_from-DWI_to-MNI_xfm.npy
    prealign_file = join(afq_derivatives_dir, subject_path, f'sub-{subject}_dwi_prealign_from-DWI_to-MNI_xfm.npy')

    ## validate prealign
    # np.round(np.load(prealign_file))
    # array([[ 1.,  0., -0.,  1.],
    #        [-0.,  1., -0., -0.],
    #        [ 0., -0.,  1., -2.],
    #        [ 0.,  0.,  0.,  1.]])

    # mapping from dwi image to MNI image
    # NOTE: using prealign here not necessary
    # mapping = reg.read_mapping(mapping_file, dwi_img, MNI_T2_img, np.linalg.inv(np.load(prealign_file)))
    mapping = reg.read_mapping(mapping_file, dwi_img, MNI_T2_img)

    ## validate mapping
    # np.shape(mapping.forward) -- (193, 229, 193, 3)
    # np.shape(mapping.backward) -- (193, 229, 193, 3)

    ###############################################################################
    # move subject streamlines into MNI space

    # took along time to realize needed to move to vox (was getting no delta)
    tractogram.to_vox()

    # order of transforms is important
    # TODO: combine into single transform operation? (linear algebra operation)
    sl_xform = tractogram.streamlines
    sl_xform = dtu.transform_tracking_output(sl_xform, dwi_img.affine) # voxel to mm scaner
    sl_xform = dtu.transform_tracking_output(sl_xform, np.linalg.inv(MNI_T2_img.affine)) # mm scanner to voxel in MNI
    sl_xform = list(dtu.transform_tracking_output(sl_xform, np.linalg.inv(np.load(prealign_file)))) # apply prealignment in MNI

    # assume in same coordinate space
    delta = dts.values_from_volume(mapping.forward, sl_xform, np.eye(4))

    ## validiate delta 
    # check delta values exist (needed to convert to VOX)
    # np.count_nonzero([np.count_nonzero(delta[i]) for i in range(len(delta))])

    moved_sl = [d + s for d, s in zip(delta, sl_xform)]

    sft = StatefulTractogram(moved_sl, MNI_T2_img, Space.VOX)

    return sft


def get_density_map(tractogram):
    import numpy as np
    from dipy.io.utils import create_nifti_header, get_reference_info
    import dipy.tracking.utils as dtu
    import nibabel as nib

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = dtu.density_map(tractogram.streamlines, np.eye(4), vol_dims)
    # force to unsigned 8-bit
    tractogram_density = np.uint8(tractogram_density)
    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    return density_map_img


def get_dice_coefficients(source_sft, target_sft):
    from AFQ.utils.volume import dice_coeff

    
    source_sft.to_vox()
    source_map = get_density_map(source_sft)

    target_sft.to_vox()
    target_map = get_density_map(target_sft)

    return dice_coeff(source_map, target_map)


def load_MNI_clusters():
    from os.path import join, exists
    from dipy.io.streamline import load_tractogram, save_tractogram

    tractograms = {}

    for subject in subjects:
        tractograms[subject] = []
        subbundle_base_dir = join('subbundles', experiment_name, bundle_name, subject, data_dir)
        
        for cluster_id in range(K):
            MNI_tractogram_file = join(subbundle_base_dir, f'{subject}_{bundle_name}_{cluster_id}_MNI.trk')

            # load subject clusters into MNI space
            if not exists(MNI_tractogram_file):
                tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
                tractogram_file = join(subbundle_base_dir, tractogram_name)
                
                if not exists(tractogram_file):
                    print(f'{subject} {bundle_name} {cluster_id} not found')
                    continue

                tractogram = load_tractogram(tractogram_file, 'same')

                sft = move_tractogram_to_MNI_space(data_dir, subject, tractogram)

                ## Save tractogram to visually inspect in MI Brain
                # NOTE: remember to Globally Reinit MI Brain (otherwise streamlines may not be loaded correctly)

                # TODO: some streamlines appears outside the brain boundary in the MNI template image
                save_tractogram(sft, MNI_tractogram_file, bbox_valid_check=False)
            else:
                sft = load_tractogram(MNI_tractogram_file, 'same')

            tractograms[subject].append(sft)
    
    return tractograms


def match_clusters():
    # calculate dice coefficeints for each permutation
    import itertools

    tractograms = load_MNI_clusters()

    # consider generating an adjacency matrix

    # set first subject as the target
    for target_subject in subjects:
        for source_subject in [subject for subject in subjects if subject != target_subject]:
            for (source_tractogram, target_tractogram), (source_cluster_id, target_cluster_id) in zip(itertools.product(tractograms[source_subject], tractograms[target_subject]), itertools.product(range(len(tractograms[source_subject])), range(len(tractograms[target_subject])))):
                print(source_subject, source_cluster_id, target_subject, target_cluster_id, get_dice_coefficients(source_tractogram, target_tractogram))


###############################################################################
# NOTE: begin small; compare three subjects and use majority vote to relabel cluster
# These subjects were initailly chosen randomly from the 44 subjects in test-restest
# and their data has been downloaded locally

subjects = ['125525', '175439', '562345']

data_dir = 'HCP_1200'
experiment_name = 'HCP_test_retest_MASE_CSD'
model_name = 'mase_fa_r2_is_mdf'
bundle_name = 'SLF_L'
K = 3


# load subject bundle into MNI space
# for subject in subjects:
#     # TODO: if MNI tractogram exists load it.
#     # load subject streamlines
#     subbundle_base_dir = join('subbundles', experiment_name, bundle_name, subject, data_dir)

#     # ./subbundles/HCP_test_retest_MASE_CSD/SLF_L/125525/HCP_1200/SLF_L.trk
#     tractogram_file = join(subbundle_base_dir, f'{bundle_name}.trk')
#     tractogram = load_tractogram(tractogram_file, 'same')

    # # validate tractogram
    # tractogram.space -- <Space.RASMM: 'rasmm'>

    # tractogram.space_attributes -- get everything all at once (affine, dimensions, voxel_sizes, voxel_order)
    # tractogram.dimensions -- compare to dwi_img.shape
    # array([145, 174, 145], dtype=int16)
    # tractogram.affine -- compare to dwi_img.affine
    # array([[  -1.25,    0.  ,    0.  ,   90.  ],
    #        [   0.  ,    1.25,    0.  , -126.  ],
    #        [   0.  ,    0.  ,    1.25,  -72.  ],
    #        [   0.  ,    0.  ,    0.  ,    1.  ]], dtype=float32)

    # tractogram.streamlines
    # len(tractogram.streamlines) -- 1290
    # np.shape(tractogram.streamlines)

    # each streamline i has different length N => np.shape(tractogram.streamlines[i]) == (N, 3)
    # np.shape(tractogram.streamlines.data)
    # tractogram.streamlines.common_shape
    # tractogram.streamlines.total_nb_rows
    # tractogram.streamlines.data.size -- common_shape*total_nb_rows

#     sft = move_tractogram_to_MNI_space(data_dir, subject, tractogram)

#     ## Save tractogram to visually inspect in MI Brain
#     # NOTE: remember to Globally Reinit MI Brain (otherwise streamlines may not be loaded correctly)

#     # TODO: some streamlines appears outside the brain boundary in the MNI template image
#     save_tractogram(sft, join(subbundle_base_dir, f'{subject}_{bundle_name}_MNI.trk'), bbox_valid_check=False)

###############################################################################
# load streamline cluster labels
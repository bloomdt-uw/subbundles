

subjects = [
    '103818', '105923', '111312', '114823', '115320',
    '122317', '125525', '130518', '135528', '137128',
    '139839', '143325', '144226', '146129', '149337',
    '149741', '151526', '158035', '169343', '172332',
    '175439', '177746', '185442', '187547', '192439',
    '194140', '195041', '200109', '200614', '204521',
    '250427', '287248', '341834', '433839', '562345',
    '599671', '601127', '627549', '660951', # '662551', 
    '783462', '859671', '861456', '877168', '917255'
]

sessions = ['HCP_1200', 'HCP_Retest']

# from os.path import exists, join
# for subject in subjects:
#     for session in sessions:
#         for cluster in ['0', '1', '2']:
#             f_name = join('./subbundles/HCP_test_retest/SLF_L/', subject, session, f'mase_fa_r2_is_mdf_cluster_{cluster}.trk')
#             if exists(f_name):
#                 clean(f_name)


def clean(tractogram_filename):
    from os.path import splitext
    from dipy.io.streamline import load_tractogram, save_tractogram
    from dipy.io.stateful_tractogram import StatefulTractogram
    from AFQ.segmentation import clean_bundle
    
    tractogram = load_tractogram(tractogram_filename, 'same')
    clean_tractogram = clean_bundle(tractogram)
    sft = StatefulTractogram.from_sft(clean_tractogram.streamlines, tractogram)
    # sft.to_vox()

    base, ext = splitext(tractogram_filename)

    save_tractogram(sft, base+'_clean'+ext, False)

def union(test_tractogram_filename, retest_tractogram_filename):
    from dipy.io.streamline import load_tractogram, save_tractogram

    test_tractogram = load_tractogram(test_tractogram_filename, 'same')
    retest_tractogram = load_tractogram(retest_tractogram_filename, 'same')

    # todo fdata find overlap
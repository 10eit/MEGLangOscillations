import mne
import numpy as np
import os.path as op
from mne_bids import BIDSPath
from bids import BIDSLayout
from mne.coreg import Coregistration
from mne.io import read_info


bids_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids'
layout=BIDSLayout(bids_root)
subject_list=layout.get_subjects()
subject_list.remove('emptyroom')
subject_list.remove('FYL20250106ZSX')

sessions=['00', '01']
runs={'00':['1', '2'], '01':['1','2','3']}
tasks={'00':'rest', '01':'semvstr'}
subjects=subject_list
meg_suffix = 'meg'
ann_suffix = 'ann'
ica_suffix = 'ica'
epo_suffix = 'epo'
task = 'semvstr'

deriv_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives'
preproc_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives/preprocessing'
fs_root = op.join(bids_root, 'derivatives/FreeSurfer')
epoch_root = op.join(deriv_root,'methods/pattern_classification')

print("Sanity Check:", len(subject_list))

for subject in subject_list:
    session=sessions[0]
    print(subject)
    fname_raw = BIDSPath(subject=subject, session=session,
                        task=tasks[session], run=f'0{runs[session][0]}', suffix=meg_suffix, datatype='meg',
                        root=bids_root, extension='.fif', check=False)
    info = read_info(fname_raw)
    subjects_dir=fs_root
    plot_kwargs = dict(
        subject=subject,
        subjects_dir=fs_root,
        surfaces="head-dense",
        dig=True,
        eeg=[],
        meg="sensors",
        show_axes=True,
        coord_frame="meg",
    )
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
    fiducials = "auto"  # get fiducials from fsaverage
    coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials, on_defects='ignore')
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=10, nasion_weight=5.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )

    trans_path=BIDSPath(root=preproc_root, subject=subject, session=session, datatype='meg', suffix='trans',extension='.fif', check=False)
    print(trans_path)
    mne.write_trans(trans_path, coreg.trans, overwrite=True)
    trans_path=trans_path.update(session=sessions[1])
    mne.write_trans(trans_path, coreg.trans, overwrite=True)
    del coreg

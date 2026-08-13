import os
import mne

from mne_bids import (
    BIDSPath,
    read_raw_bids
)
import mne
import bids

empty_layout=bids.BIDSLayout('/home/fyl/ProjSemvstr/bids_dataset/meg_bids')
empty_sessions=empty_layout.get_sessions()
empty_sessions.remove('00')
empty_sessions.remove('01')
print(empty_sessions)

for ses in empty_sessions:
    empty_file = BIDSPath(root='/home/fyl/ProjSemvstr/bids_dataset/meg_bids', session=ses, subject='emptyroom', datatype='meg', suffix='meg', task='noise', extension='.fif')
    empty_fif = read_raw_bids(empty_file).load_data().copy()
    empty_fif = empty_fif.notch_filter([50,100],n_jobs=12,verbose='error').filter(1,45,n_jobs=12,verbose='error')
    noise_cov=mne.compute_raw_covariance(empty_fif, verbose=False)
    cov_fname = BIDSPath(root='/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives/preprocessing', session=ses, subject='emptyroom', datatype='meg', suffix='cov', task='noise',description= ('1to45MEG'),extension='.fif', check=False)
    os.makedirs(os.path.dirname(cov_fname.fpath), exist_ok=True)
    noise_cov.save(cov_fname, overwrite=True)
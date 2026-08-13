import mne
import os.path as op
from bids import BIDSLayout
from mne_bids import BIDSPath, read_raw_bids

n_jobs = 24  
bids_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids'
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
fs_root = op.join(bids_root, 'derivatives/FreeSurfer')

layout = BIDSLayout(bids_root)
subject_list = layout.get_subjects()
exclude_subjects = {'FYL20240117ZX', 'FYL20240703ZZX', 'FYL20250106ZSX', 'emptyroom'}
subject_list = [s for s in subject_list if s not in exclude_subjects]
print(f"Number of subjects: {len(subject_list)}")

for subject in subject_list:
    raw_path = BIDSPath(
        subject=subject, session='01', task='semvstr',
        datatype='meg', root=bids_root, suffix='meg', run='1', check=False
    )
    print(raw_path)
    raw = read_raw_bids(raw_path, verbose=True)
    raw.pick('meg')
    info = raw.info

    model = mne.make_bem_model(subject, ico=4, conductivity=(0.3,), subjects_dir=fs_root)
    bem = mne.make_bem_solution(model)
    src = mne.setup_source_space(subject, subjects_dir=fs_root, surface='white', spacing = 'ico4',
                                    add_dist=False, n_jobs=n_jobs)
    trans_file = BIDSPath(subject=subject, session='01', suffix='trans',
                            datatype='meg', root=preproc_root, extension='.fif', check=False)
    fwd = mne.make_forward_solution(info, trans=trans_file.fpath, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=n_jobs)
    emptyroom_session = subject[3:11]
    cov_path = BIDSPath(subject='emptyroom', session=emptyroom_session, task='noise',
                        description='1to45MEG', suffix='cov', datatype='meg',
                        root=preproc_root, extension='.fif', check=False)
    noise_cov = mne.read_cov(cov_path.fpath)
    inverse_path = BIDSPath(subject=subject, session='01',
                task='semvstr', suffix='inv', datatype='meg',description='1to45MEG',
                root=preproc_root, extension='.fif', check=False)

    inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, fixed = True, rank = 'info')
    print(inv)
    mne.minimum_norm.write_inverse_operator(inverse_path, inv, overwrite=True)

print("All Done!!!")
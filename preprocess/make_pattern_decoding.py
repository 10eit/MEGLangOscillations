import os
import mne
from mne_bids import BIDSPath
from bids import BIDSLayout
import pandas as pd
import numpy as np
import os.path as op

bids_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids'
layout=BIDSLayout(bids_root)

sessions=['00', '01']
runs={'00':['1', '2'], '01':['1','2','3']}
tasks={'00':'rest', '01':'semvstr'}
meg_suffix = 'meg'
ann_suffix = 'ann'
ica_suffix = 'ica'
epo_suffix = 'epo'
task = 'semvstr'

reject = dict(grad=5000e-13,    # T / m (gradiometers)
              mag=5e-12,        # T (magnetometers)
             )
session=sessions[1]
task=tasks[session]

layout = BIDSLayout(bids_root)
subject_list = layout.get_subjects()
exclude_subjects = {'FYL20240117ZX', 'FYL20240703ZZX', 'FYL20250106ZSX', 'emptyroom'}
subject_list = [s for s in subject_list if s not in exclude_subjects]
print(f"Number of subjects: {len(subject_list)}")


for subject in subject_list[-2:]:
    deriv_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives'
    preproc_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives/preprocessing'
    os.makedirs(op.join(deriv_root,'methods/pattern_classification'),exist_ok=True)
    epoch_root = op.join(deriv_root,'methods/pattern_classification')
    deriv_file = BIDSPath(subject=subject, session=session,
                    task=task, suffix=epo_suffix, datatype='meg',
                    root=epoch_root, extension='.fif', check=False)
    print(deriv_file)
    epoch_list=[]
    for run in runs[session]:
        ## define the data of each run. (after ICA)
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, suffix=ica_suffix, datatype='meg',
                    root=preproc_root, extension='.fif', check=False)
        print(bids_path)
    
        # define the bids path to raw data and events file
        raw_path = bids_path.copy().update(root=bids_root, 
                    suffix=meg_suffix, extension='.fif', check=False)
        # print(raw_path.fpath)
        beh_path=BIDSPath(subject=subject, session=session, task=task, run=f'0{run}', datatype='beh', suffix='beh', root=bids_root, extension='.tsv', check=True)
        if subject.startswith('FYL2025'):
            beh_data=pd.read_csv(beh_path)
        else:
            beh_data=pd.read_csv(beh_path)

        abb = (beh_data['encoding'].isin(['ABB_run1', 'ABB_run2'])) & (beh_data['word_deviant']=='n') & ~(beh_data['response'].isin([1, -1]))
        aab = (beh_data['encoding'].isin(['AAB_run1', 'AAB_run2'])) & (beh_data['word_deviant']=='n') & ~(beh_data['response'].isin([1, -1]))
        aabb = (beh_data['encoding'].isin(['AABB_run'])) & (beh_data['word_deviant']=='n') & ~(beh_data['response'].isin([1, -1]))
        abab = (beh_data['encoding'].isin(['ABAB_run'])) & (beh_data['word_deviant']=='n') & ~(beh_data['response'].isin([1, -1]))
    
        raw = mne.io.read_raw_fif(bids_path, preload=True).notch_filter([50, 100],n_jobs=24).filter(1,45,n_jobs=24)
        
        # Now pick events
        events_encoding=mne.find_events(raw, stim_channel='STI001')
        events_auditory=mne.find_events(raw, stim_channel='STI002')
        events_delay=mne.find_events(raw, stim_channel='STI003')
        events_response=mne.find_events(raw, stim_channel='STI004')
        events_encoding[:,2]=1
        events_auditory[:,2]=2
        events_delay[:,2]=4
        events_response[:,2]=8
        
        events=np.vstack([events_auditory,
                          events_delay,
                          events_response])
        
        onset_correction = beh_data['correction_times'].tolist()
        print(max(onset_correction))
        delays = np.array(onset_correction)
        events_tmp=events_auditory.copy()
    
        sfreq = raw.info['sfreq']
        print(sfreq)

        # Convert delay times from seconds to samples
        delay_samples = (np.array(delays) * sfreq).astype(int)
        
        # Adjust the start_time_point by adding the delay in samples
        corrected_events = events_tmp.copy()
        corrected_events[:, 0] += delay_samples
        
        corrected_events[abb,2]=501
        corrected_events[aab,2]=502
        corrected_events[abab,2]=503
        corrected_events[aabb,2]=504
    
        trials_mapping={'abb':501,'aab':502,'abab':503,'aabb':504}
        
        epochs = mne.Epochs(raw,
                corrected_events, trials_mapping,
                tmin=-1.0, tmax=2.5,
                baseline=None,
                proj=True,
                picks = 'all',
                detrend = 1,
                reject=reject,
                reject_by_annotation=True,
                on_missing = 'warn',
                preload=True)
        epoch_list.append(epochs)
    
    # Concatenate them together
    session_epoch=mne.concatenate_epochs(epoch_list, on_mismatch='ignore')
    
    print("Epoch Component:\n")
    print(session_epoch)

    if not os.path.exists(deriv_file.directory):
        os.makedirs(deriv_file.directory)
    session_epoch.save(deriv_file, overwrite=True)
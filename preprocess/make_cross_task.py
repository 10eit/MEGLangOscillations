import os.path as op
import os
import pandas as pd
import numpy as np

import mne
from mne_bids import BIDSPath
from bids import BIDSLayout

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

subject_list=layout.get_subjects()
subject_list.remove('emptyroom')
### Subjects with Bad Behavioral Outcomes.
subject_list.remove('FYL20240117ZX')
subject_list.remove('FYL20240703ZZX')
subject_list.remove('FYL20250106ZSX')

print(f"Number of Subjects: {len(subject_list)}")

for subject in subject_list:
    deriv_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives'
    preproc_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids/derivatives/preprocessing'
    os.makedirs(op.join(deriv_root,'methods/cross_task'),exist_ok=True)
    epoch_root = op.join(deriv_root,'methods/cross_task')
    deriv_file = BIDSPath(subject=subject, session=session,
                    task=task, suffix=epo_suffix, datatype='meg',
                    root=epoch_root, extension='.fif', check=False)
    print(deriv_file)
    epoch_list=[]
    for run in runs[session]:
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, suffix=ica_suffix, datatype='meg',
                    root=preproc_root, extension='.fif', check=False)
        print(bids_path)
        raw_path = bids_path.copy().update(root=bids_root, 
                    suffix=meg_suffix, extension='.fif', check=False)
        beh_path=BIDSPath(subject=subject, session=session, task=task, run=f'0{run}', datatype='beh', suffix='beh', root=bids_root, extension='.tsv', check=True)
        if subject.startswith("FYL2025"):
            beh_data=pd.read_csv(beh_path)
            print(beh_data)
        else:
            beh_data=pd.read_csv(beh_path)

        ## Only Epoch Correct Trials
        three_str = (beh_data['length'] == 3) & (beh_data['word_deviant']=='str') & (beh_data['response'] == 1)
        three_sem = (beh_data['length'] == 3) & (beh_data['word_deviant']=='sem') & (beh_data['response'] == -1)
        three_ambg_str = (beh_data['length'] == 3) & (beh_data['word_deviant']=='both') & (beh_data['response'] == 1)
        three_ambg_sem = (beh_data['length'] == 3) & (beh_data['word_deviant']=='both') & (beh_data['response'] == -1)

        four_str = (beh_data['length'] == 4) & (beh_data['word_deviant']=='str') & (beh_data['response'] == 1)
        four_sem = (beh_data['length'] == 4) & (beh_data['word_deviant']=='sem') & (beh_data['response'] == -1)
        four_ambg_str = (beh_data['length'] == 4) & (beh_data['word_deviant']=='both') & (beh_data['response'] == 1)
        four_ambg_sem = (beh_data['length'] == 4) & (beh_data['word_deviant']=='both') & (beh_data['response'] == -1)

        raw = mne.io.read_raw_fif(bids_path, preload=True).notch_filter([50, 100],n_jobs=24).filter(1,45,n_jobs=24)

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

        delay_samples = (np.array(delays) * sfreq).astype(int)
        
        # Adjust the start_time_point by adding the delay in samples
        corrected_events = events_tmp.copy()
        corrected_events[:, 0] += delay_samples
        
        corrected_events[three_str,2]=501
        corrected_events[three_sem,2]=502
        corrected_events[three_ambg_str,2]=503
        corrected_events[three_ambg_sem,2]=504

        corrected_events[four_str,2]=601
        corrected_events[four_sem,2]=602
        corrected_events[four_ambg_str,2]=603
        corrected_events[four_ambg_sem,2]=604

        trials_mapping={'three_str':501, 'three_sem':502, 'three_ambg_str':503, 'three_ambg_sem': 504,
                        'four_str':601, 'four_sem':602, 'four_ambg_str':603, 'four_ambg_sem': 604}
        
        epochs = mne.Epochs(raw,
                corrected_events, trials_mapping,
                tmin=-1.0, tmax=1.8,
                baseline=None,
                proj=True,
                picks = 'all',
                detrend = 1,
                reject=reject,
                reject_by_annotation=True,
                on_missing = 'warn',
                preload=True)
        epoch_list.append(epochs)
    
    session_epoch=mne.concatenate_epochs(epoch_list, on_mismatch='ignore')

    print("Epoch Component:\n")
    print(session_epoch)

    if not os.path.exists(deriv_file.directory):
        os.makedirs(deriv_file.directory)
    session_epoch.save(deriv_file, overwrite=True)
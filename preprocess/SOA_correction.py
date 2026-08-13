import os.path as op
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids
from bids import BIDSLayout

bids_root = '/home/fyl/ProjSemvstr/bids_dataset/meg_bids'
layout=BIDSLayout(bids_root)
sessions=['00', '01']
runs={'00':['1', '2'], '01':['1','2','3']}
tasks={'00':'rest', '01':'semvstr'}

session=sessions[1]
task=tasks[session]

subject_list=layout.get_subjects()
subject_list.remove('emptyroom')

### Subjects with Bad Behavioral Outcomes.
subject_list.remove('FYL20240117ZX')
subject_list.remove('FYL20240228BS')
subject_list.remove('FYL20240703ZZX')

path = 'onset_correction.csv'
correction = pd.read_csv(path,index_col=False)

def add_delay(subject):
    print(f'Initializing and adding delay to {subject}')
    for run in range(1, 4):
        # 构建文件路径
        beh_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            run=f'0{run}',
            datatype='beh',
            suffix='beh',
            root=bids_root,
            extension='.tsv',
            check=True
        )
        
        # 加载行为数据
        beh_data = pd.read_csv(beh_path)
        
        # 移除所有 `correction_times` 列
        if 'correction_times' in beh_data.columns:
            beh_data = beh_data.drop(columns=['correction_times'])
            print(f'Run {run}: Existing correction_times column removed.')

        # 合并数据
        merged_df = pd.merge(
            beh_data,
            correction[['word', 'time']],
            left_on='word_stim',
            right_on='word',
            how='left'
        )

        # 重命名并更新列
        merged_df = merged_df.rename(columns={'time': 'correction_times'})
        merged_df = merged_df.drop(columns=['word'])

        # 保存更新后的数据
        merged_df.to_csv(beh_path, index=False)
        print(f'Run {run} data updated with new correction_times!')

    return f"{subject} delay reinitialized and added successfully!"

for subject in subject_list[-3:]:
    add_delay(subject)
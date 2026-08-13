# Preprocessing

BIDS-format MEG 预处理流水线脚本（从仓库根目录与 `Preprocessing/` 目录整理而来，
脚本内容保持原样，仅做归类）。建议按编号顺序运行。

| 脚本                       | 作用                                                      |
| -------------------------- | --------------------------------------------------------- |
| `make_transfile.py`        | 生成 BIDS head-position / trans 文件                       |
| `make_cov.py`              | 由 empty-room 数据估计 noise covariance (1–45 Hz notch+滤波) |
| `make_inv.py`              | 计算 inverse operator (eLORETA / FreeSurfer BEM)          |
| `SOA_correction.py`        | 修正 deviant/standard 的 SOA 延迟标注                      |
| `make_epochs_decoding.py`  | 抽取 decoding 用 epochs 并存盘                              |
| `make_epochs_deviants_all.py` | 抽取所有 deviant 条件的 epochs                          |
| `make_epochs_rsa.py`       | 抽取 RSA 用 epochs                                          |
| `make_epochs_conflict.py`  | 抽取 STR vs SEM 冲突条件 epochs                            |
| `make_pattern_decoding.py` | 抽取 pattern decoding 用 epochs                            |
| `make_cross_task.py`       | 合并 / 对齐跨 task (rest vs semvstr) epochs                |
| `make_epoch_pattern.py`    | 抽取 pattern-decoding 的 epochs（备用版本）                 |

## 目录约定
所有脚本默认使用：

```
bids_root    = .../bids_dataset/meg_bids
preproc_root = bids_root/derivatives/preprocessing   # cov, inv, trans
fs_root      = bids_root/derivatives/FreeSurfer       # FreeSurfer subjects
derivatives  = bids_root/derivatives/methods/<分析名>  # epochs 输出
```

替换脚本顶部的 `bids_root` 路径即可适配到新机器 / 新 repo。
排除被试（行为数据不良）的清单在各脚本内的 `exclude_subjects` 集合里。
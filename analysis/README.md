# Analysis toolkit (`kit`)

`kit` is a small, consolidated MEG analysis toolkit提炼自本项目散落在各 Figure 文件夹
里的可复用分析逻辑。它把以下能力收敛成可 import 的函数：

| 模块            | 能力                                                          |
| --------------- | ------------------------------------------------------------ |
| `kit.io`        | BIDS / derivatives 读取 epochs、inverse operator、fsaverage src |
| `kit.source`    | source space 工具：morph、label time-course、medial wall、adjacency、Schaefer 标注 |
| `kit.power`     | **ERF / spectral power**：sensor Morlet TFR、source induced power、频段平均 (theta/alpha/beta/gamma)、baseline logratio |
| `kit.decoding`  | **各类 decoding**：sliding/diagonal decoding、**temporal generalization** (GeneralizingEstimator)、cross-task generalization、permutation surrogate、LogisticRegression + StandardScaler pipeline |
| `kit.searchlight` | **searchlight decoding** 与 **searchlight RSA**：graph-distance searchlight ball、并行逐顶点解码、Spearman RSA |
| `kit.rdm`       | **计算 RDM**：Euclidean RDM、**Crossnobis RDM** (LedoitWolf + k-fold cross)、crossnobis RDM movie (rsatoolbox, noise precision from residuals)、balanced k-fold、RDM <-> vector |
| `kit.rsa`       | **RSA**：Spearman 相关、双模型逐被试 RSA + paired t-test、组水平 RSA 检验 |
| `kit.stats`     | 组水平统计：against-chance one-sample t、FDR 校正、permutation p-value、显著性 mask |

## 安装 / 使用

`kit` 是纯 Python 包，无需安装：把 `analysis/` 放在 `PYTHONPATH` 上即可。

```python
from kit import io, power, decoding, rdm, rsa, searchlight, stats
from kit.decoding import make_logistic_clf, temporal_generalization
from kit.rdm import crossnobis_rdm, euclidean_rdm, crossnobis_rdm_movie
```

## 典型流程

### 1) Spectral power
```python
from kit.power import compute_sensor_power, compute_source_power
band_power = compute_sensor_power(epochs[cond], baseline=(2.4, 2.7),
                                  crop=(-0.1, 1.8), n_jobs=24)
src_power, verts = compute_source_power(epochs[cond], inv, morph=morph,
                                        baseline=(2.4, 2.7), crop=(-0.1, 1.8))
```

### 2) Temporal generalization decoding
```python
from kit.decoding import make_logistic_clf, temporal_generalization
clf = make_logistic_clf(C=1.0)
scores = temporal_generalization(X, y, clf=clf, scoring="roc_auc", n_jobs=40)
# 训练集 vs 测试集 (cross-task generalization, e.g. Figure6):
scores = temporal_generalization(train_X, train_y, clf=clf,
                                 score_data=test_X, score_labels=test_y)
```

### 3) Searchlight decoding / RSA
```python
from kit.searchlight import searchlight_decoding, searchlight_rsa
from kit.source import get_medial_vertices, source_adjacency
medial = get_medial_vertices(fs_root, ico_sample=4)
adj = source_adjacency(fs_src)
scores = searchlight_decoding(fs_src, data, y, clf, d=1,
                              exclude_medial=medial, n_jobs=40)
rho, p = searchlight_rsa(model_rdm_vec, source_power, adj,
                         d=1, exclude_medial=medial, n_jobs=40)
```

### 4) RDM (Euclidean + Crossnobis)
```python
from kit.rdm import euclidean_rdm, crossnobis_rdm, crossnobis_rdm_movie
euc = euclidean_rdm(X, zscore=True)                      # (n_items, n_items)
xnb = crossnobis_rdm(data, labels, n_folds=5)           # per-ROI crossnobis
movie = crossnobis_rdm_movie(data, labels, times)       # (n_times, n_cond, n_cond)
```

### 5) RSA + 统计
```python
from kit.rsa import rsa_spearman, rsa_compare_models
from kit.stats import fdr_correct, cluster_significance_mask
rho, p = rsa_spearman(neural_rdm, model_rdm)
sig = cluster_significance_mask(tg_scores, chance=0.5, alpha=0.001)
```

> 注：`kit` 只做分析与统计；绘图未纳入，留在 `Publication/visualization`。
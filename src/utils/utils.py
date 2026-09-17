
import torch
import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)


def clipdata2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image': batch[4].cuda(),
        'clip_image': batch[5].cuda(),
        'clip_text': batch[6].cuda(),
        'multi_category': batch[7].cuda(),
    }
    return batch_data


def data2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image': batch[4].cuda(),
    }
    return batch_data


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def _round4(x):
    """sklearn 返回 float / ndarray，统一为 Python float 并保留 4 位小数。"""
    if isinstance(x, np.ndarray):
        x = float(x.item()) if x.size == 1 else float(x.mean())
    return round(float(x), 4)


def _binarize_preds(y_pred):
    return np.around(np.asarray(y_pred, dtype=float)).astype(int)


def _safe_auc(y_true, y_score):
    try:
        return _round4(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_classification_scores(y_true, y_pred_bin):
    """单领域二分类指标；样本过少或仅单类时 zero_division=0 避免异常。"""
    kwargs = dict(zero_division=0)
    if len(set(y_true)) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'fscore': 0.0,
            'acc': _round4(accuracy_score(y_true, y_pred_bin)),
        }
    return {
        'precision': _round4(precision_score(y_true, y_pred_bin, average='binary', **kwargs)),
        'recall': _round4(recall_score(y_true, y_pred_bin, average='binary', **kwargs)),
        'fscore': _round4(f1_score(y_true, y_pred_bin, average='binary', **kwargs)),
        'acc': _round4(accuracy_score(y_true, y_pred_bin)),
    }


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}

    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {'y_true': [], 'y_pred': []}

    for i, c in enumerate(category):
        c = reverse_category_dict[int(c)]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    y_pred_bin = _binarize_preds(y_pred)

    for c, res in res_by_category.items():
        cat_metrics = _safe_classification_scores(
            res['y_true'],
            _binarize_preds(res['y_pred']),
        )
        auc = _safe_auc(res['y_true'], res['y_pred'])
        if auc is not None:
            cat_metrics['auc'] = auc
        metrics_by_category[c] = cat_metrics

    auc_global = _safe_auc(y_true, y_pred)
    if auc_global is not None:
        metrics_by_category['auc'] = auc_global

    global_kwargs = dict(zero_division=0)
    metrics_by_category['metric'] = _round4(
        f1_score(y_true, y_pred_bin, average='macro', **global_kwargs)
    )
    metrics_by_category['recall'] = _round4(
        recall_score(y_true, y_pred_bin, average='macro', **global_kwargs)
    )
    metrics_by_category['precision'] = _round4(
        precision_score(y_true, y_pred_bin, average='macro', **global_kwargs)
    )
    metrics_by_category['acc'] = _round4(accuracy_score(y_true, y_pred_bin))

    return metrics_by_category


def metricsTrueFalse(y_true, y_pred, category, category_dict):
    result = metrics(y_true, y_pred, category, category_dict)

    y_gt = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_pred, dtype=float)

    fake = {}
    real = {}
    thresh_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    realnews_TP, realnews_TN = [0] * 9, [0] * 9
    realnews_FP, realnews_FN = [0] * 9, [0] * 9
    fakenews_TP, fakenews_TN = [0] * 9, [0] * 9
    fakenews_FP, fakenews_FN = [0] * 9, [0] * 9
    realnews_sum, fakenews_sum = [0] * 9, [0] * 9

    for thresh_idx, thresh in enumerate(thresh_list):
        y_bin = (y_score >= thresh).astype(int)
        for idx in range(len(y_bin)):
            if y_gt[idx] == 1:
                fakenews_sum[thresh_idx] += 1
                if y_bin[idx] == 0:
                    fakenews_FN[thresh_idx] += 1
                    realnews_FP[thresh_idx] += 1
                else:
                    fakenews_TP[thresh_idx] += 1
                    realnews_TN[thresh_idx] += 1
            else:
                realnews_sum[thresh_idx] += 1
                if y_bin[idx] == 1:
                    realnews_FN[thresh_idx] += 1
                    fakenews_FP[thresh_idx] += 1
                else:
                    realnews_TP[thresh_idx] += 1
                    fakenews_TN[thresh_idx] += 1

    real_precision, fake_precision = [0] * 9, [0] * 9
    real_recall, fake_recall = [0] * 9, [0] * 9
    real_F1, fake_F1 = [0] * 9, [0] * 9

    for thresh_idx in range(9):
        real_precision[thresh_idx] = realnews_TP[thresh_idx] / max(
            1, realnews_TP[thresh_idx] + realnews_FP[thresh_idx]
        )
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(
            1, fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx]
        )
        real_recall[thresh_idx] = realnews_TP[thresh_idx] / max(
            1, realnews_TP[thresh_idx] + realnews_FN[thresh_idx]
        )
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(
            1, fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx]
        )
        real_F1[thresh_idx] = (
            2 * real_recall[thresh_idx] * real_precision[thresh_idx]
        ) / max(1, real_recall[thresh_idx] + real_precision[thresh_idx])
        fake_F1[thresh_idx] = (
            2 * fake_recall[thresh_idx] * fake_precision[thresh_idx]
        ) / max(1, fake_recall[thresh_idx] + fake_precision[thresh_idx])

    fake['precision'] = fake_precision[0]
    fake['recall'] = fake_recall[0]
    fake['F1'] = fake_F1[0]
    real['precision'] = real_precision[0]
    real['recall'] = real_recall[0]
    real['F1'] = real_F1[0]
    result['real'] = real
    result['fake'] = fake
    return result


class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print('curent', self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print('Max', self.max)

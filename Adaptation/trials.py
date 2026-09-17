"""受控 trial 和余弦分数，对应 ipynb 的 controlled_pairs。"""
import numpy as np
from Adaptation.standardize import l2_normalize


def controlled_pairs(labels, nontargets_per_anchor=20, max_target_pairs_per_speaker=0, seed=20260705):
    """生成 target / non-target pair 列表。

    - 同说话人之间的所有 pair 作为 target
    - 每个 anchor 最多取 nontargets_per_anchor 个 non-target
    """
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels)

    target_pairs = []
    for speaker in sorted(set(labels.tolist())):
        indices = np.where(labels == speaker)[0]
        pairs = [(int(indices[a]), int(indices[b])) for a in range(len(indices) - 1) for b in range(a + 1, len(indices))]
        if max_target_pairs_per_speaker > 0 and len(pairs) > max_target_pairs_per_speaker:
            chosen = rng.choice(len(pairs), size=max_target_pairs_per_speaker, replace=False)
            pairs = [pairs[int(i)] for i in np.sort(chosen)]
        target_pairs.extend(pairs)

    non_target_pairs = []
    seen = set()
    for i in range(len(labels)):
        candidates = np.where(labels != labels[i])[0]
        if len(candidates) == 0:
            continue
        k = int(nontargets_per_anchor)
        if k <= 0 or k >= len(candidates):
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=k, replace=False)
        for j in chosen:
            j = int(j)
            pair = (i, j) if i < j else (j, i)
            if pair not in seen:
                seen.add(pair)
                non_target_pairs.append(pair)

    return target_pairs, non_target_pairs


def scores_from_embeddings(embeddings, target_pairs, non_target_pairs):
    """用 L2 归一化 embedding 的余弦相似度计算 trial 分数。"""
    z = l2_normalize(embeddings)
    scores, labels = [], []
    for i, j in target_pairs:
        scores.append(float(np.dot(z[i], z[j])))
        labels.append(1)
    for i, j in non_target_pairs:
        scores.append(float(np.dot(z[i], z[j])))
        labels.append(0)
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32)

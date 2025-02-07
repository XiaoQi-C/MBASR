import numpy as np
import random
import math


def cal_prob(length, a=0.8):

    item_indices = np.arange(length)
    # Items towards the end of the sequence will have higher importance
    item_importance = np.power(a, length - item_indices)

    total = np.sum(item_importance)
    prob = item_importance / total
    return prob


def cal_prob2(length, a=0.8):

    item_indices = np.arange(length)
    # Items towards the start of the sequence will have higher importance
    item_importance = np.power(a, item_indices)

    total = np.sum(item_importance)
    prob = item_importance / total
    return prob


def cal_prob3(length, selected_subseq_index, gamma):
    
    subseq_indices = np.arange(length)
    # Items closer to the selected item will have higher importance.
    subseq_importance = np.power(gamma, abs(subseq_indices - selected_subseq_index))

    total = np.sum(subseq_importance)
    prob = subseq_importance / total
    return prob



def item_del(seq, behavior, beta):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)  # The number of subsequences
    index = np.arange(num_sub_seq)
    sub_prob = cal_prob(num_sub_seq)[::-1]
    num_samples = math.ceil(num_sub_seq * beta)
    num_samples = random.sample(range(num_samples + 1), k=1)[0]
    if num_samples == 0:
        del_item_seq = np.concatenate(seq_)
        del_behavior_seq = np.concatenate(behavior_)
        return del_item_seq, del_behavior_seq
    else:
        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            num_sampled_sub_seq = len(sub_seq)
            # The last purchased item cannot be deleted
            sub_seq_behavior = behavior_[index]
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_seq_behavior)
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase
            if num_sampled_sub_seq_click == 0 or num_sampled_sub_seq == 1:
                pass
            else:
                del_index = random.sample(range(num_sampled_sub_seq_click), k=1)[0]
                deled_sub_seq = np.delete(sub_seq, del_index)
                deled_sub_behavior = np.delete(sub_seq_behavior, del_index)
                seq_[index] = deled_sub_seq
                behavior_[index] = deled_sub_behavior

        del_item_seq = np.concatenate(seq_)
        del_behavior_seq = np.concatenate(behavior_)
        # del_length = len(del_item_seq)

        return del_item_seq, del_behavior_seq



def item_reorder(seq, behavior, alpha):
    seq_ = seq.copy()
    behavior_ = behavior.copy()

    num_sub_seq = len(seq_)  # 子序列的个数
    index = np.arange(num_sub_seq)
    sub_prob = cal_prob(num_sub_seq)[::-1]

    num_samples = math.ceil(num_sub_seq * alpha)
    num_samples = random.sample(range(num_samples + 1), k=1)[0]

    if num_samples == 0:
        reorder_item_seq = np.concatenate(seq_)
        reorder_behavior_seq = np.concatenate(behavior_)
        return reorder_item_seq, reorder_behavior_seq
    else:
        # 按照概率获得m * beta个子序列的index
        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            sub_behavior = behavior_[index]
            num_sampled_sub_seq = len(sub_seq)
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_behavior) # purchase=1
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase
            item_index = np.arange(num_sampled_sub_seq_click)
            if num_sampled_sub_seq_click == 0:
                continue
            item_prob = cal_prob(num_sampled_sub_seq_click)[::-1]
            if len(item_prob) == 1:
                continue
            indices = np.random.choice(item_index, p=item_prob, size=num_sampled_sub_seq_click, replace=False)
            if len(indices) == 1:
                continue
            shuffled_sub_seq = np.array(sub_seq)[indices].tolist() + sub_seq[num_sampled_sub_seq_click:]
            shuffled_sub_behavior = np.array(sub_behavior)[indices].tolist() + sub_behavior[num_sampled_sub_seq_click:]

            seq_[index] = shuffled_sub_seq
            behavior_[index] = shuffled_sub_behavior

        reorder_item_seq = np.concatenate(seq_)
        reorder_behavior_seq = np.concatenate(behavior_)

        return reorder_item_seq, reorder_behavior_seq



def sub_sequence_reorder(seq, behavior, gamma):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)  # 子序列的个数
    index = np.arange(num_sub_seq)

    # 首先随机选择个子序列索引
    selected_item1_index = random.sample(range(num_sub_seq), k=1)[0]

    prob = cal_prob3(num_sub_seq, selected_item1_index,gamma)

    # 按照概率prob随机选择第二个子序列索引：
    selected_item2_index = np.random.choice(index, p=prob, size=1, replace=False)[0]
    # 交换两个子序列：
    seq_[selected_item1_index], seq_[selected_item2_index] = seq_[selected_item2_index], seq_[selected_item1_index]
    behavior_[selected_item1_index], behavior_[selected_item2_index] = behavior_[selected_item2_index], behavior_[
        selected_item1_index]

    reorder_sub_seq = np.concatenate(seq_)
    reorder_sub_behavior = np.concatenate(behavior_)

    return reorder_sub_seq, reorder_sub_behavior



# 插入购买物品作为浏览
def item_insert(seq, behavior, length, max_len, zeta):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)  # 子序列的个数

    index = np.arange(num_sub_seq)
    sub_prob = cal_prob(num_sub_seq)[::-1]
    num_samples = math.ceil(num_sub_seq * zeta) # 向上取整

    num_samples = random.sample(range(num_samples + 1), k=1)[0]

    if num_samples == 0 or length + num_samples > max_len:
        insert_item_seq = np.concatenate(seq_)
        insert_behavior_seq = np.concatenate(behavior_)
        return insert_item_seq, insert_behavior_seq
    else:
        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            sub_seq_behavior = behavior_[index]

            num_sampled_sub_seq = len(sub_seq)
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_seq_behavior)
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase
            if num_sampled_sub_seq_click == 0 or num_sampled_sub_seq == 1:
                pass
            else:

                click_indices = np.arange(num_sampled_sub_seq_click)  # 创建从 0 到 n-1 的索引张量

                item_importance = np.power(0.8, num_sampled_sub_seq_click - click_indices)

                total = np.sum(item_importance)
                prob = item_importance / total

                chosen_index = np.random.choice(click_indices, size=1, replace=False, p=prob)[0]
                # 在行为序列和物品序列的相应位置插入元素
                sub_seq_behavior.insert(chosen_index + 1, 0)
                sub_seq.insert(chosen_index + 1, sub_seq[-1])

                seq_[index] = sub_seq
                behavior_[index] = sub_seq_behavior

        insert_item_seq = np.concatenate(seq_)
        insert_behavior_seq = np.concatenate(behavior_)

        return insert_item_seq, insert_behavior_seq



# 插入相似子序列- 超过长度就不插入-----效果较好
def add_similar_user_interactions(item_sequences_1, behavior_sequences_1,item_sequences_2, behavior_sequences_2, max_seq_len, length, k):

        sub_length_1 = len(item_sequences_1)
        sub_length_2 = len(item_sequences_2)

        # 取后端
        prob_1 = cal_prob(sub_length_1, a=0.8)
        prob_2 = cal_prob(sub_length_2, a=0.8)

        index_1 = np.arange(sub_length_1)
        index_2 = np.arange(sub_length_2)

        if item_sequences_1 and item_sequences_2:
            length_ = length

            # 使用超参控制抽取子序列个数
            cnt = sub_length_1 * k
            index_num = math.ceil(cnt)
            # 抽取相似用户的待插入子序列
            index_selected = np.random.choice(index_2, p=prob_2, size=index_num, replace=True)

            for i in range(index_num):
                # 将item_sequences_2的某子序列插入到item_sequences_1的某位置
                if length_ + len(item_sequences_2[index_selected[i]]) <= max_seq_len:
                    # 确定要插入的位置
                    index_to_insert = np.random.choice(index_1, p=prob_1, size=1, replace=False)
                    sub_length_1 += 1
                    index_1 = np.arange(sub_length_1)
                    prob_1 = cal_prob(sub_length_1, a=0.8)

                    item_sequences_1.insert(index_to_insert[0], item_sequences_2[index_selected[i]])
                    behavior_sequences_1.insert(index_to_insert[0], behavior_sequences_2[index_selected[i]])
                    length_ += len(item_sequences_2[index_selected[i]])

        aug_seq = np.concatenate(item_sequences_1)
        aug_behavior = np.concatenate(behavior_sequences_1)

        return aug_seq, aug_behavior


def augmentation(items, behavios, lengths, item_num, max_seq_len, jaccard_similarity, item_seq_grouped, behavior_seq_grouped, tag, alpha, beta, gamma, lamda, zeta, k, p):
    batch_size = len(items)
    aug_items = []
    aug_behaviors = []
    aug_lengths = []

    for i in range(batch_size):
        item_seq = items[i]
        behavior_seq = behavios[i]
        length = lengths[i]

        unpad_item_seq = np.array(item_seq)[:length]
        unpad_behavior_seq = np.array(behavior_seq)[:length]

        mask = (unpad_behavior_seq[:-1] == 1) & (unpad_behavior_seq[1:] == 0)

        split_indices = np.where(mask)[0] + 1
        split_indices = np.insert(split_indices, 0, 0)
        split_indices = np.append(split_indices, length)
        item_sequences = [unpad_item_seq[start:end].tolist() for start, end in
                          zip(split_indices[:-1], split_indices[1:])]
        behavior_sequences = [unpad_behavior_seq[start:end].tolist() for start, end in
                              zip(split_indices[:-1], split_indices[1:])]

        if tag == 1:
            aug_seq, aug_behavior = item_reorder(item_sequences, behavior_sequences, alpha)

        elif tag == 2:
            aug_seq, aug_behavior = item_del(item_sequences, behavior_sequences, beta)

        elif tag == 3:
            aug_seq, aug_behavior = item_insert(item_sequences, behavior_sequences, length, max_seq_len, zeta)

        elif tag == 4:
            aug_seq, aug_behavior = sub_sequence_reorder(item_sequences, behavior_sequences, gamma)

        elif tag == 5:
            # 生成一个0到1之间的随机数
            random_value = random.random()

            if random_value < p:
                # 按照相似度抽取相似用户的seq
                similarity = np.array(jaccard_similarity[i])

                total = np.sum(similarity)
                prob = similarity / total

                # 根据概率抽取索引
                index = np.arange(len(jaccard_similarity[i]))
                sim_user_selected = np.random.choice(index, p=prob, size=1, replace=False)
                sim_user_selected = sim_user_selected[0]

            else:
                # 抽取随机一个相似用户的seq
                sim_user_selected = random.randint(0, len(item_seq_grouped[i]) - 1)

            random_item_seq = item_seq_grouped[i][sim_user_selected]
            random_behavior_seq = behavior_seq_grouped[i][sim_user_selected]
            length_2 = len(random_item_seq)

            unpad_item_seq_2 = np.array(random_item_seq)[:length_2]
            unpad_behavior_seq_2 = np.array(random_behavior_seq)[:length_2]

            mask = (unpad_behavior_seq_2[:-1] == 1) & (unpad_behavior_seq_2[1:] == 0)

            split_indices = np.where(mask)[0] + 1
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, length_2)
            item_sequences_2 = [unpad_item_seq_2[start:end].tolist() for start, end in
                                zip(split_indices[:-1], split_indices[1:])]
            behavior_sequences_2 = [unpad_behavior_seq_2[start:end].tolist() for start, end in
                                    zip(split_indices[:-1], split_indices[1:])]

            aug_seq, aug_behavior = add_similar_user_interactions(item_sequences, behavior_sequences,
                                                                    item_sequences_2, behavior_sequences_2,
                                                                    max_seq_len,
                                                                    length, k)
        elif tag == 6:
            # 生成一个0到1之间的随机数
            random_value = random.random()

            if random_value < p:
                # 按照相似度抽取相似用户的seq
                similarity = np.array(jaccard_similarity[i])

                total = np.sum(similarity)
                prob = similarity / total

                # 根据概率抽取索引
                index = np.arange(len(jaccard_similarity[i]))
                sim_user_selected = np.random.choice(index, p=prob, size=1, replace=False)
                sim_user_selected = sim_user_selected[0]

            else:
                # 抽取随机一个相似用户的seq
                sim_user_selected = random.randint(0, len(item_seq_grouped[i]) - 1)

            random_item_seq = item_seq_grouped[i][sim_user_selected]
            random_behavior_seq = behavior_seq_grouped[i][sim_user_selected]
            length_2 = len(random_item_seq)

            unpad_item_seq_2 = np.array(random_item_seq)[:length_2]
            unpad_behavior_seq_2 = np.array(random_behavior_seq)[:length_2]

            mask = (unpad_behavior_seq_2[:-1] == 1) & (unpad_behavior_seq_2[1:] == 0)

            split_indices = np.where(mask)[0] + 1
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, length_2)
            item_sequences_2 = [unpad_item_seq_2[start:end].tolist() for start, end in
                                zip(split_indices[:-1], split_indices[1:])]
            behavior_sequences_2 = [unpad_behavior_seq_2[start:end].tolist() for start, end in
                                    zip(split_indices[:-1], split_indices[1:])]

            aug_seq, aug_behavior = add_similar_user_interactions(item_sequences, behavior_sequences,
                                                                    item_sequences_2, behavior_sequences_2,
                                                                    max_seq_len,
                                                                    length, k)

            mask = (aug_behavior[:-1] == 1) & (aug_behavior[1:] == 0)

            length = len(aug_seq)

            split_indices = np.where(mask)[0] + 1
            split_indices = np.insert(split_indices, 0, 0)
            split_indices = np.append(split_indices, length)
            item_sequences = [aug_seq[start:end].tolist() for start, end in
                              zip(split_indices[:-1], split_indices[1:])]
            behavior_sequences = [aug_behavior[start:end].tolist() for start, end in
                                  zip(split_indices[:-1], split_indices[1:])]

            # 再进行子序列间扰动
            aug_seq, aug_behavior = sub_sequence_reorder(item_sequences, behavior_sequences, gamma)

        else:
            # 其余不改变数据集
            aug_seq, aug_behavior = unpad_item_seq, unpad_behavior_seq


        item_seq = np.pad(aug_seq, (0, max_seq_len - len(aug_seq)), 'constant', constant_values=item_num)
        behavior_seq = np.pad(aug_behavior, (0, max_seq_len - len(aug_behavior)), 'constant', constant_values=2)

        item_seq = item_seq.tolist()
        behavior_seq = behavior_seq.tolist()

        aug_items.append(item_seq)
        aug_behaviors.append(behavior_seq)
        aug_lengths.append(len(aug_seq))

    return aug_items, aug_behaviors, aug_lengths


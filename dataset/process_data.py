import os
import pandas as pd

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union != 0 else 0

def calculate_similarity(data_loader):
    num_users = len(data_loader)
    data_loader['similar_jaccard'] = -1

    # 稀疏化item_seq列
    item_seq = data_loader['item_seq']

    # 处理item_seq_grouped列
    item_seq_grouped = data_loader['item_seq_grouped'].tolist()

    all_jaccard_similarities = []

    # 遍历每个用户
    for i in range(num_users):
        user_item_set = set(item_seq[i])
        jaccard_similarities = []
        # 获取当前用户的稀疏化相似用户序列
        sim_items_sparse = item_seq_grouped[i]

        for j, sim_item_seq in enumerate(sim_items_sparse):

            sim_item_set = set(sim_item_seq)
            item_similarity = jaccard_similarity(user_item_set, sim_item_set)
            jaccard_similarities.append(item_similarity) # 记录相似度

        all_jaccard_similarities.append(jaccard_similarities)
    # 为'similar_jaccard' 列设置值
    data_loader['similar_jaccard'] = all_jaccard_similarities

    return data_loader

if __name__ == '__main__':
    data_directory = 'datasets/UB/data'

    data_loader = pd.read_pickle(os.path.join(data_directory, 'train.df'))

    # 稀疏化原始数据集
    # 稀疏化item_seq列
    item_seq = data_loader['item_seq'].apply(lambda row: row[-5::-5][::-1])
    data_loader['item_seq'] = item_seq

    # 稀疏化behavior_seq列
    behavior_seq = data_loader['behavior_seq'].apply(lambda row: row[-5::-5][::-1])
    data_loader['behavior_seq'] = behavior_seq

    # 保存更新后的DataFrame
    data_loader.to_pickle(os.path.join(data_directory, 'train_1.pkl'))

    data_loader = pd.read_pickle(os.path.join(data_directory, 'train_1.pkl'))

    # 创建新列 'item_seq_grouped'，将每个 target 相同的 item_seq 值组合为嵌套列表
    data_loader['item_seq_grouped'] = data_loader['target'].map(
        data_loader.groupby('target')['item_seq'].apply(lambda x: [item for item in x]))

    # 创建新列 'behavior_seq_grouped'，将每个 target 相同的 behavior_seq 值组合为嵌套列表
    data_loader['behavior_seq_grouped'] = data_loader['target'].map(
        data_loader.groupby('target')['behavior_seq'].apply(lambda x: [item for item in x]))

    # 统计 'item_seq_grouped' 列中列表长度为 1 的行数
    single_element_count = data_loader[data_loader['item_seq_grouped'].apply(len) == 1].shape[0]

    print(f"无相同target的行: {single_element_count}")

    # 为每个用户计算相似用户的相似度
    data_loader = calculate_similarity(data_loader)

    # 保存更新后的DataFrame
    data_loader.to_pickle(os.path.join(data_directory, 'train_2.pkl'))

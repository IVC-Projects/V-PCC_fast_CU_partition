partition_list = ['CU_VERT_SPLIT', 'CU_HORZ_SPLIT', 'CU_QUAD_SPLIT', 'CU_TRIV_SPLIT', 'CU_TRIH_SPLIT', 'NO_SPLIT']

import os


def quchong(path1, path2):
    mp = {}
    with open(path1, "r") as f:
        for line in f:
            if len(line.split()) == 15 and line.split()[-1] in partition_list:
                st = line.strip()
                if st in mp:
                    mp[st] += 1
                else:
                    mp[st] = 1

    with open(path2, "w") as f:
        for i, (key, value) in enumerate(mp.items()):
            if i == len(mp) - 1:
                f.write(key)
            else:
                f.write(key + '\n')


if __name__ == '__main__':
    path = r"\VTM_result\data_statistic"
    dst_path = r'\VTM_result\data_statistic_norepeat'
    for file in os.listdir(path):
        if file.split('.')[-1] == 'txt' and len(file.split('_')) == 3:
            quchong(os.path.join(path, file), os.path.join(dst_path, file))
            print(os.path.join(path, file), os.path.join(dst_path, file))


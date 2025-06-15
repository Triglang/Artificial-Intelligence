import os
import re

def delete_numbered_files(directory='.'):   
    to_delete = []

    # 遍历目录并筛选文件
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        img_name = filename.split('.')[0]

        img_id = int(img_name.split('_')[1])
        if img_id >= 5000:
            to_delete.append(filepath)
        # # 检测test文件
        # test_match = test_pattern.match(filename)
        # if test_match:
        #     number = int(test_match.group(1))
        #     if number >= 5001:
        #         to_delete.append(filepath)
        #         continue  # 已匹配test，无需再检查training

        # # 检测training文件
        # training_match = training_pattern.match(filename)
        # if training_match:
        #     number = int(training_match.group(1))
        #     if number >= 5001:
        #         to_delete.append(filepath)

    # 执行删除操作前确认
    if not to_delete:
        print("没有找到符合条件的文件。")
        # return

    print("以下文件将被删除：")
    for f in to_delete:
        print(f" - {os.path.basename(f)}")
    
    confirm = input(f"\n确认删除以上 {len(to_delete)} 个文件吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消。")
        return

    # 执行删除
    deleted_count = 0
    for filepath in to_delete:
        try:
            os.remove(filepath)
            print(f"成功删除: {os.path.basename(filepath)}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {os.path.basename(filepath)}: {str(e)}")
    
    print(f"\n操作完成，成功删除 {deleted_count}/{len(to_delete)} 个文件。")
    
    test_img = []
    train_img = []
    # 检查 test 和 trainning 是否从 0 到 4999
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        img_name = filename.split('.')[0]
        
        img_mod = img_name.split('_')[0]
        img_id = int(img_name.split('_')[1])
        if img_mod == 'test':
            test_img.append(img_id)
        elif img_mod == 'training':
            train_img.append(img_id)

    test_img.sort()
    train_img.sort()

    print(f"\n检查结果：")
    if test_img == list(range(0, 5000)):
        print("test 文件夹从 0 到 4999 按顺序排列。")
    else:
        print("test 文件夹不完整或顺序错误。")
        
        # 输出缺失的文件
        for i in range(0, 3000):
            if i not in test_img:
                print(f" - 缺失 test_{i}.jpg")
        
    if train_img == list(range(0, 5000)):
        print("training 文件夹从 0 到 4999 按顺序排列。")
    else:
        print("training 文件夹不完整或顺序错误。")
        
        

if __name__ == "__main__":
    target_dir = input("请输入要清理的目录路径（留空为当前目录）: ").strip() or '.'
    if not os.path.exists(target_dir):
        print("错误：目录不存在！")
    else:
        delete_numbered_files(target_dir)
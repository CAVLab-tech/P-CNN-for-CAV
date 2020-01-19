# 文件放置的路径：和data文件夹同目录
import os
train_src_path = "data/train/"
train_dst_file = "./train.txt"
train_dict = ["train_car", "train_people"]

if __name__ == '__main__':
    result=[]
    for file in os.listdir(train_src_path):
        if file == train_dict[0]:
            pth = os.path.join(train_src_path, file)
            for image in os.listdir(pth):
                name = os.path.join(file, image)
                result.append('/home/cqx/caffe/models/VGGnet/data/train/'+name+' 0\n')
        # elif file == train_dict[1]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 1\n')
        # elif file == train_dict[2]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 2\n')
        # elif file == train_dict[3]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 3\n')
        # elif file == train_dict[4]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 4\n')
        # elif file == train_dict[5]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 5\n')
        # elif file == train_dict[6]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 6\n')
        # elif file == train_dict[7]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 7\n')
        # elif file == train_dict[8]:
        #     pth = os.path.join(train_src_path, file)
        #     for image in os.listdir(pth):
        #         name = os.path.join(file, image)
        #         result.append(name+' 8\n')
        else:
            pth = os.path.join(train_src_path, file)
            for image in os.listdir(pth):
                name = os.path.join(file, image)
                result.append('/home/cqx/caffe/models/VGGnet/data/train/'+name+' 1\n')
    txtfile = open(train_dst_file, "w")
    txtfile.writelines(result)
    txtfile.close()
    print("done!")
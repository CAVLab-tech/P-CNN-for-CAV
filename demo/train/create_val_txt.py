# 文件放置的路径：和data文件夹同目录
import os

val_src_path = "data/val/"
val_dst_file = "./val.txt"

train_dict = ["testingcar", "testingpeople"]
if __name__ == '__main__':
    result=[]
    for file in os.listdir(val_src_path):
        name = file.split(sep='_')
        if name[0] == train_dict[0]:
            result.append('/home/cqx/caffe/models/VGGnet/data/val/'+file+' 0\n')
        # elif name[0] == train_dict[1]:
        #     result.append(file+' 1\n')
        # elif name[0] == train_dict[2]:
        #     result.append(file+' 2\n')
        # elif name[0] == train_dict[3]:
        #     result.append(file+' 3\n')
        # elif name[0] == train_dict[4]:
        #     result.append(file+' 4\n')
        # elif name[0] == train_dict[5]:
        #     result.append(file+' 5\n')
        # elif name[0] == train_dict[6]:
        #     result.append(file+' 6\n')
        # elif name[0] == train_dict[7]:
        #     result.append(file+' 7\n')
        # elif name[0] == train_dict[8]:
        #     result.append(file+' 8\n')
        else:
            result.append('/home/cqx/caffe/models/VGGnet/data/val/'+file+' 1\n')
    txtfile = open(val_dst_file, "w")
    txtfile.writelines(result)
    txtfile.close()
    print ("done!")
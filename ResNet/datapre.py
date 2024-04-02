from sklearn.model_selection import train_test_split
from random import sample

class balance:
    def __init__(self,img1_list,img0_list):
        self.img1_list = img1_list
        self.img0_list = img0_list

    def balance_no(self):
        return self.img1_list,self.img0_list

    def balance_downsample(self):
        normal_num = len(self.img0_list)
        tumor_num = len(self.img1_list)
        minn = min(normal_num,tumor_num)
        normal = sample(self.img0_list,minn)
        tumor = sample(self.img1_list,minn)
        return tumor,normal

    def balance_oversample(self):
        normal_num = len(self.img0_list)
        tumor_num = len(self.img1_list)
        mmax = max(normal_num,tumor_num)
        if normal_num>tumor_num:
            normal = self.img0_list
            tumor_a = normal_num//tumor_num
            tumorlist = self.img1_list*tumor_a
            tumorrest = sample(self.img1_list,normal_num-tumor_a*tumor_num)
            tumor = tumorlist + tumorrest
        else:
            tumor = self.img1_list
            normal_a = tumor_num//normal_num
            normallist = self.img0_list*normal_a
            normalrest = sample(self.img0_list,tumor_num-normal_a*normal_num)
            normal = normallist + normalrest
        return tumor,normal
'''
a = list(range(15,40))
b = list(range(5))

f = balance(a,b)
print(f.balance_no())
'''
# def balance_downsample(img1_list, img0_list):
#     normal_num = len(img0_list)
#     tumor_num = len(img1_list)
#     minn = min(normal_num,tumor_num)
#     normal = sample(img0_list,minn)
#     tumor = sample(img1_list,minn)
#     return tumor,normal
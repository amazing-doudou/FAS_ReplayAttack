# -*- coding: utf-8 -*- 
import os 
import numpy as np
import random
import shutil 

def main():
    
    #copy_error()

   write_list()
   # split_list()
   
   #filter_list()

def filter_list():

    f1 = open('list/test1203_modi.txt','r')
    list1 = f1.readlines()

    f2 = open('list/test1209_shorter.txt','w')

    for i in range(len(list1)):
        if list1[i].find('jiguang') == -1 and list1[i].find('youmo') == -1 and list1[i].find('xizhen') == -1 and list1[i].find('juanqu') == -1 and list1[i].find('close') == -1:
            # print(list1[i])
            # exit()
            #for i in range(4):
            f2.write(list1[i])


def copy_error():
    f1 = open('errorlist_se48.txt','r')
    list1 = f1.readlines()

    root = '/mnt/sda1/error_48/'

    for i in range(len(list1)):
        folder_path = root+str(i)+'/'

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        path = list1[i].split('**')
        folders = get_folder(path)
        for j in range(len(folders)-1):
           
            shutil.copyfile(path[j+1].strip('\n'), root+str(i)+'/'+str(i)+'_' + folders[j+1])

def get_folder(pathlist):
    folders = []
    for i in range(len(pathlist)):
        folders.append(pathlist[i].split('/')[-1].strip('\n'))
    
    return folders

def split_list():
    f1 = open('list/people_list.txt','r')
    f2 = open('list/pic_list.txt','r')
    f3_ = open('list/people_kuangshi_list.txt','r')

    f3 = open('list/train.txt','w')
    f4 = open('list/test.txt','w')

    split(f1,f3,f4)
    split(f2,f3,f4)
    split(f3_,f3,f4)


def split(orig,train,test):
    list1 = orig.readlines()
    for i in range(len(list1)):

        a = random.random()
        if a > 0.1:
            train.write(list1[i])
        else:
            test.write(list1[i])


def write_list():
    
    root_folder = '/mnt/sda1/bmp_crop/RGB/bmp/'
    list = open('list/20200720_error.txt','w')

    folder_ = ''
    count = 0
    wpath = ''
    filelist = os.walk(root_folder)
    for fpath,dirs,fs in filelist:
        for f in fs:
            path = (os.path.join(fpath,f))
            
            #and (path.find('juanqu')== -1) and (path.find('jiguang')== -1)  and (path.find('youmo')== -1) and (path.find('xiezhen')== -1) 
            
           # if path.find('/2019_')!= -1  and (path.find('.bmp')!= -1) :
            #if path.find('/2019_')!= -1  and (path.find('.bmp')!= -1) and path.find('_2019') == -1:
            if path.find('/2019_') == -1  and (path.find('.bmp')!= -1) and path.find('_2019') != -1:
           # if path.find('x22019') != -1  and (path.find('.bmp')!= -1):
                
            #if path.find('/2019_')!= -1  and (path.find('.bmp')!= -1) and path.find('_2019') == -1: #and path.find('_2019') == -1
                folder = check_folder(path)
                
                if folder_ != '':
                    if folder_ == folder:
                        if count < 2:
                            wpath = wpath + '**' + path
                            count = count + 1
                        else:
                            wpath = wpath + '**' + path
                            

                            list.write(wpath + ' 0\n')
                            
                            count = 0
                            wpath = ''
                    else:
                        folder_ = folder
                        # if count == 2:
                        #     list.write(wpath + ' 0\n')
                        

                        count = 0
                        wpath = ''
                else:
                    folder_ = folder
                    wpath = wpath + '**' + path
                    count = count + 1


def  check_folder(path):
    tpath = path.split('/')
    folder = ''
    

    for i in range(len(tpath)):
        
        #if tpath[i].find('2019_') != -1 :
        #if tpath[i].find('/2019_') != -1 and path.find('_2019') == -1:
        #if tpath[i].find('x22019') != -1:

        if tpath[i].find('/2019_')== -1 and path.find('_2019') != -1: #and tpath[i].find('_2019') == -1
            
            folder = tpath[i]
            n = 1
            while(tpath[i+n].find('.bmp')== -1):
                folder = folder + '/' + tpath[i+n]
                n = n + 1

            # if tpath[i+1].find('.bmp') == -1:
                
            #     folder = tpath[i]+'/' + tpath[i+1]
            # else:
            #     folder = tpath[i]
            break

    return folder                


    
if __name__ == '__main__':
    main()
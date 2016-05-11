import os
from shutil import copyfile


train_list = '../training_set_files'
test_list = '../testing_set_files.txt'

rel_data = '../../data/relation_data'
rel_data_train = '../../data/relation_data_train'
rel_data_test = '../../data/relation_data_test'

coref_data = '../../data/coref_data'
coref_data_train = '../../data/coref_data_train'
coref_data_test = '../../data/coref_data_test'

# read the train and test list and copy the corresponding json files from original
# folders to the train and test folders

if not os.path.isdir(coref_data_train):
	os.mkdir(coref_data_train)

if not os.path.isdir(coref_data_test):
	os.mkdir(coref_data_test)

if not os.path.isdir(rel_data_train):
	os.mkdir(rel_data_train)

if not os.path.isdir(rel_data_test):
	os.mkdir(rel_data_test)
	
# create the training data 
fr = open(train_list,'rb')
tot_files = 0
rel_missing = 0
coref_missing = 0
for line in fr:
    #print line.rstrip()
    tot_files += 1
    src1 = rel_data+'/'+line.rstrip()
    dst1 = rel_data_train+'/'+line.rstrip()
    src2 = coref_data+'/'+line.rstrip()
    dst2 = coref_data_train+'/'+line.rstrip()
    if os.path.exists(src1):
        copyfile(src1,dst1)
    else:
        print 'file does not exit - relations: '+ line.rstrip()
        rel_missing += 1 
    
    if os.path.exists(src2):
        copyfile(src2,dst2)
    else:
        print 'file does not exit - coref: '+ line.rstrip()    
        coref_missing += 1 

fr.close()
print 'total files in training list = '+str(tot_files)
print 'missing files in relations data = '+str(rel_missing)
print 'missing files in coref data = '+str(coref_missing)
print '-------------------------------------------------'
# create the testing data 
fs = open(test_list,'rb')
tot_files = 0
rel_missing = 0
coref_missing = 0
for line in fs:
    #print line.rstrip()
        
    if 'apf.xml.json' in line.rstrip():
        liness = line.rstrip().split('.apf.xml.json')
        tot_files += 1
        src1 = rel_data+'/'+liness[0]+'.json'
        dst1 = rel_data_test+'/'+liness[0]+'.json'
        src2 = coref_data+'/'+liness[0]+'.json'
        dst2 = coref_data_test+'/'+liness[0]+'.json'    
        if os.path.exists(src1):
            copyfile(src1,dst1)
        else:
            print 'file does not exit - relations: '+ liness[0]+'.json'
            rel_missing += 1 
        
        if os.path.exists(src2):
            copyfile(src2,dst2)
        else:
            print 'file does not exit - coref: '+ liness[0]+'.json'    
            coref_missing += 1 

fs.close()
print 'total files in testing list = '+str(tot_files)
print 'missing files in relations data = '+str(rel_missing)
print 'missing files in coref data = '+str(coref_missing)


import codecs


labelInfo = "/Users/1110732/PycharmProjects/CS492I_teamUDA/CV/fashion_data2/train/train_label"

with codecs.open(labelInfo,'r',encoding='utf-8') as fr:
    lines = fr.read().strip().split("\n")


fileNMlabelDict = {}
for line in lines[1:]:
    _, label, fileNMinfo = line.split()
    fileNMlabelDict[fileNMinfo] = label

print (fileNMlabelDict)

import glob, os

for level in ['easy','med','med2','hard']:
    fileNMlist = glob.glob(os.path.join("/Users/1110732/PycharmProjects/CS492I_teamUDA/CV/trainingData",level,'*.jpg'))

    baseNMlist = [os.path.basename(fileNM) for fileNM in fileNMlist]
    labelList = [int(fileNMlabelDict[fileNM]) for fileNM in baseNMlist]

    print (level)
    print (len(baseNMlist),baseNMlist)
    print (len(labelList),labelList)
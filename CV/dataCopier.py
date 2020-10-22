import shutil, glob
from CV.dataInfo import labeledData_filenms
import os


srcFileList = glob.glob("/Users/1110732/PycharmProjects/CS492I_teamUDA/CV/fashion_data2/train/train_data/*.jpg")


for targetFile in srcFileList:
    baseNM = os.path.basename(targetFile)

    if baseNM in labeledData_filenms:
        shutil.copy2(targetFile, os.path.join("/Users/1110732/PycharmProjects/CS492I_teamUDA/CV/trainingData",baseNM))

print (len(srcFileList))
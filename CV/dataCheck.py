

from dataInfo import labeledData_filenms, labeledData_labels
from pseudoDataInfo import pseudo_data_names, pseudo_label

import codecs, ast

with codecs.open("./pseudoInfo.txt",'r',encoding='utf-8') as fr:
    lines = fr.read().strip().split("\n")


FileNM_list = ast.literal_eval(lines[0])
Classes_list = ast.literal_eval(lines[1])

if FileNM_list == (labeledData_filenms + pseudo_data_names): print ("YES")
if Classes_list == (labeledData_labels + pseudo_label): print ("YES")
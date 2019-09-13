import glob

directory_map={
    "train":{
        "0":"./data/train/0/*.jpg",
        "1":"./data/train/1/*.jpg"
    },
    "validate":{
        "0":"./data/validate/0/*.jpg",
        "1":"./data/validate/1/*.jpg"
    }
}

files={
    "train":"./data/train/data.txt",
    "validate":"./data/validate/data.txt"
}

for data_type in directory_map:
    text=""
    for class_no in directory_map[data_type]:
        list=glob.glob(directory_map[data_type][class_no])
        for l in list:
            text+="%s %s\n"%(l,class_no)
    with open(files[data_type], mode="w") as f:
        f.write(text)

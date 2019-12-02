import os
def recurse(src,pattern,list_of_all_files,querycams,lowhigh):
    if os.path.isfile(src):
        test=src.split('.')[-1]
        if test in pattern:
            list_of_all_files.append(src)
        return
    children=os.listdir(src)
    if src.split('/')[-1] in querycams:
        for child in children:
            if int(child.split('_')[1])>=lowhigh[0] and int(child.split('_')[1])<=lowhigh[1]:
                recurse(src+'/'+child,pattern, list_of_all_files,querycams,lowhigh)
    else:
        for child in children:
            recurse(src+'/'+child,pattern, list_of_all_files,querycams,lowhigh)
    return
def datacollector(src, querycams=[],lowhigh=(0,100)):
    list_of_all_files=[]
    recurse(src,['png','jpg','jpeg'],list_of_all_files,querycams,lowhigh)
    return list_of_all_files
    
def idcollector(globalImgMapping):
    globalIdSet=set()
    idtoImgMappingDict={}
    for img_path in globalImgMapping: 
        person=img_path.split('/')[-2]
        id=int(person.split('_')[1]) 
        if id not in globalIdSet:
            globalIdSet.add(id)
            idtoImgMappingDict[id]=[img_path]
        else:
            idtoImgMappingDict[id].append(img_path)
    return globalIdSet,idtoImgMappingDict
def getNextIdset(globalIdList, NumberofIdsInIdSet=5):
    if len(globalIdList)<NumberofIdsInIdSet:
        return globalIdList,[]
    return globalIdList[:NumberofIdsInIdSet],globalIdList[NumberofIdsInIdSet:]
def extractImgMappingFromIds(idtoImgMappingDict,idSet):
    img_mapping=[]
    for id in idSet:
        img_mapping.extend(idtoImgMappingDict[id])
    return img_mapping

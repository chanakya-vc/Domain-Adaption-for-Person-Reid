import os
def recurse(src,pattern,list_of_all_files,querycams,images_per_index={}):
    if os.path.isfile(src):
        test=src.split('.')[-1]
        if test in pattern:
            list_of_all_files.append(src)
            person=src.split('/')[-2]
            index=person.split('_')[1]
            
#            print(index)
#            print(src)
            index=int(index)
            
            if index in images_per_index:
                images_per_index[index]+=1
            else:
                images_per_index[index]=1
        return
    children=os.listdir(src)
    if src.split('/')[-1] in querycams:
        for child in children:
            if int(child.split('_')[1])<=200:
                recurse(src+'/'+child,pattern, list_of_all_files,querycams,images_per_index)
    else:
        for child in children:
            recurse(src+'/'+child,pattern, list_of_all_files,querycams,images_per_index)
    return
def datacollector(src, querycams=[]):
    list_of_all_files=[]
    images_per_index={}
    recurse(src,['png','jpg','jpeg'],list_of_all_files,querycams,images_per_index)
    return list_of_all_files, images_per_index
    
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

src=r'C:\Research\datasets\prid_2011\multi_shot\cam_a'
t1,t2=datacollector(src,['cam_a'])
import synapseclient 
 
syn = synapseclient.Synapse() 
syn.login('zg78','Stark1998103') 

# Obtain a pointer and download the data 
syn22006407 = syn.get(entity='syn22006407', downloadLocation='/usr/xtmp/zg78/stanford_dataset') 

# Get the path to the local copy of the data file 
filepath = syn22006407.path 
print(filepath, flush=True)


syn22006006 = syn.get(entity='syn22006006', downloadLocation='/usr/xtmp/zg78/stanford_dataset') 

# Get the path to the local copy of the data file 
filepath = syn22006006.path 
print(filepath, flush=True)


syn22006404 = syn.get(entity='syn22006404', downloadLocation='/usr/xtmp/zg78/stanford_dataset') 

# Get the path to the local copy of the data file 
filepath = syn22006404.path 
print(filepath, flush=True)
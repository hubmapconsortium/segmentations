#### Segmentation wrappers around cellpose and deepcell

### Command line argument

**`--method`**    segmentation method `cellpose` or `deepcell`     
**`--dataset_dir`**     path to directory with images                
**`--gpus`**    comma separated ids of gpus to use, e.g. `"0,1,2"`, default `"all"`

### Usage 
`python main.py --dataset_dir /path/to/dataset/ --method deepcell (or cellpose) --gpus "0,1"`

### Expected **`dataset_dir`** structure:
```
dataset_dir
|-- data_dir1
|    |-- prefix1_nucleus.tif
|    `-- prefix1_cell.tif
|
|-- data_dir2
|    |-- prefix2_nucleus.tif  
|    `-- prefix2_cell.tif
|
...
| 
`-- data_dirN
     |-- prefixN_nucleus.tif  
     `-- prefixN_cell.tif      
```

### Output structure
```
dataset_dir
|-- data_dir1
|    `-- prefix1_mask.ome.tiff
|    
|-- data_dir2
|    `-- prefix2_mask.ome.tiff 
|    
...
| 
`-- data_dirN
     `-- prefixN_mask.ome.tiff     
```



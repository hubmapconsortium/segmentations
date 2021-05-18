#### Segmentation wrappers around cellpose and deepcell

Usage `python bin/main.py --dataset_dir /path/to/dataset/ --method cellpose (or deepcell)  --batch_size 10`

Expected dataset dir structure:
```
dataset_dir
|-- data_dir1
|    |-- prefix_1_nucleus.tif
|    `-- prefix_1_cell.tif
|
|-- data_dir2
|    |-- prefix_2_nucleus.tif  
|    `-- prefix_2_cell.tif
|
...
| 
`-- data_dirN
    |-- prefix_N_nucleus.tif  
    `-- prefix_N_cell.tif      
```

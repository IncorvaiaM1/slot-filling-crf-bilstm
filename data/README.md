# ATIS data (local, excluded from Git)

The prepared local folder contains the original working data. Git excludes it.
For a fresh clone, obtain the standard ATIS split used by MiuLab SlotGated-SLU
under the dataset provider's terms and place these files here:

`atis/{train,valid,test}/{seq.in,seq.out,label}`

Expected sizes: 4,478 train, 500 validation, 893 test utterances.
The loader maps `dev` to the `valid` directory. See data_loader.py.

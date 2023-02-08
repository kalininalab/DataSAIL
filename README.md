# DataSAIL: Data Splitting Against Information Leaking 



## Usage

```shell
python3 datasail.py -i <path_to_fasta_database> -o <directory_for_outputfiles>
```

Additional optional parameters:\
`-s` : clustering steps (int) (default=`4`)\
`-f` : additional fasta output (boolean flag)\
`-tr` : size of training set (default=`60`)\
`-te` : size of test set (default=`30`)\

Output:

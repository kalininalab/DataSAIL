How to add a new Similarity or distance metric to DataSAIL?

On the example of DIAMOND {links}, we demonstrate how to add a new similarity or distance metric to DataSAIL.

The DIAMOND similarity metric is a similarity metric for comparing two sets of items. It is based on the Jaccard similarity
coefficient, which is a measure of similarity between two sets. The Jaccard similarity coefficient is defined as the
size of the intersection divided by the size of the union of the two sets. The DIAMOND similarity metric is defined as
1 minus the Jaccard similarity coefficient. The DIAMOND similarity metric is used to compare two sets of items, such as
two sets of genes, two sets of proteins, or two sets of metabolites.

0. Create a fork of the DataSAIL repository

Bla bla bla

1. Installability

When adding a new similarity or distance metric to DataSAIL, make sure, it is installable from Conda

2. Registration

Register the new similarity or distance metric at various locations in the `settings.py` file.
a.
    ```python
    # Define the names of the algorithms
    ...
    DIAMOND = "diamond"
    ...
    ```
b.
    ```python
    # List of all available algorithms
    SIM_ALGOS = [WLK, MMSEQS, MMSEQS2, MMSEQSPP, FOLDSEEK, CDHIT, CDHIT_EST, ECFP, DIAMOND, ]
    ...
    ```
c.
    ```python
    # Check if the tools are installed
    INSTALLED = {
        ...
        DIAMOND: shutil.which("diamond") is not None,
        ...
    }
    ```
d.
    ```python
    # Define the names of the YAML files
    YAML_FILE_NAMES = {
        MMSEQS: "args/mmseqs2.yaml",
        MMSEQS2: "args/mmseqs2.yaml",
        MMSEQSPP: "args/mmseqspp.yaml",
        CDHIT: "args/cdhit.yaml",
        CDHIT_EST: "args/cdhit_est.yaml",
        DIAMOND: "args/diamond.yaml",
        FOLDSEEK: "args/foldseek.yaml",
        ECFP: "args/.yaml",
        MASH: "args/mash.yaml",
        MASH_SKETCH: "args/mash_sketch.yaml",
        MASH_DIST: "args/mash_dist.yaml",
    }
    ```

3. Python Implementation

Next, we create the file `diamond.py` in the `datasail/cluster` directory. This new file must contain a
`run_<tool_name>` function, which is used to compute the similarity or distance metric. The `run_<tool_name>` function
must take the following arguments:

* dataset: A DataSet-object storing the data to be clustered.
* threads: The number of threads to use for the computation.
* log_dir: An optional path to a directory where log files can be written.

The documentation of the `run_<tool_name>` function must be written in the docstring of the function but can be copied
from other metric-implementations as they are very similar.

    ```python
    def run_diamond(dataset: DataSet, threads: int, log_dir: Optional[Path] = None) -> None:
        """
        Run Diamond on a dataset in clustering mode.

        Args:
            dataset: Dataset to be clustered.
            threads: Number of threads to be used by the clustering algorithm.
            log_dir: Directory to store the logs.
        """
    ```

Then, we have to check if the tool, we want to use is installed on the system. This can be done by using the `shutil.which` function.

    ```python
        if not INSTALLED[DIAMOND]:
            raise ValueError("DIAMOND is not installed.")
    ```

Next, we have to extract potential user arguments from the `dataset` object. This is done using the `MultiYAMLParser`
class. Here, the number of parameters to be extracted is 2, as the DIAMOND tool has two stages from a FASTA file to the
similarity matrix. If the tool has only one stage, only one set of parameters has to be extracted (see `foldseek.py`).

    ```python
        parser = MultiYAMLParser(DIAMOND)
        makedb_args = parser.get_user_arguments(dataset.args, [], 0)
        blastp_args = parser.get_user_arguments(dataset.args, [], 1)
    ```

Then, we need to extract the sequences from the database into a FASTA file as only a FASTA file can be passed on to
DIAMOND.

    ```python
        extract_fasta(dataset)
    ```

Now, we construct the command line call to DIAMOND. This includes creating a folder for the tmp-files created by
DIAMOND, the actual call to DIAMOND and piping of the output to a log file or `/dev/null`.

    ```python
        result_folder = Path("diamond_results")

        cmd = lambda x: f"mkdir {result_folder} && " \
                        f"cd {result_folder} && " \
                    f"diamond makedb --in seqs.fasta --db seqs.dmnd {makedb_args} {x} && " \
                    f"diamond blastp --db seqs.dmnd --query seqs.fasta --out alis.tsv --outfmt 6 qseqid sseqid pident --threads 8 {blastp_args} {x}"

        if log_dir is None:
            cmd = cmd("> /dev/null 2>&1")
        else:
            cmd = cmd(f">> {(Path(log_dir) / f'{dataset.get_name()}_mmseqspp.log').resolve()}")

        if result_folder.exists():
            cmd = f"rm -rf {result_folder} && " + cmd
    ```

Now, we're ready to run DIAMOND.

    ```python
        subprocess.run(cmd, shell=True, check=True)
    ```

Finally, we have to parse the output of DIAMOND. We also

    ```python
        LOGGER.info("Start DIAMOND")
        LOGGER.info(cmd)
        os.system(cmd)
    ```

As something may break during the execution of the DIAMOND tool, we have to check if the output file exists.

    ```python
        if not (result_folder / "alis.tsv").is_file():
        raise ValueError("Something went wront with DIAMOND alignment. The output file does not exist.")
    ```

Now, it's time to harvest the results of the DIAMOND tool. This is done by reading the output file into a dataframe and
converting it into a table using `df.pivot`. Then, we need to fix two details about the DIAMOND output. First, the
TSV-file has no column names, so we have to add them. Second, the similarity score is computed as pident, which is
f_ident * 100. To correct this and scale the similarities back to [0,1], we have to divide p_ident by 100.

    ```python
        df = pd.read_csv(result_folder / "seqs.tsv", sep="\t")
        df.columns = ["query", "target", "pident"]
        df["fident"] = df["pident"] / 100
        table = df.pivot(index="query", columns="target", values="fident").fillna(0).to_numpy()
    ```

After deleting the temporary files, we can store the results in the `dataset` object.

    ```python
        shutil.rmtree(result_folder)

        dataset.cluster_names = dataset.names
        dataset.cluster_map = {n: n for n in dataset.names}
        dataset.cluster_similarity = table
    ```

4. Registration -- cont'd

Now, we have to add the new similarity or distance metric to the `run` function in the `cluster.py` file.

    ```python
        def similarity_clustering(dataset: DataSet, threads: int = 1, log_dir: Optional[str] = None) -> None:
        """
        ...
        """
        if dataset.similarity_algorithm == WLK:
        ...
        elif dataset.similarity.lower() == DIAMOND:
            run_diamond(dataset, threads, log_dir)
        ...
        else:
            raise ValueError(f"Unknown similarity algorithm: {dataset.similarity_algorithm}")
    ```

If you provide evidence in your upcoming pull request that the new metric is better than all other methods, you can add
your metric at the first place in the list in the `get_default` method in the settings.py file.

        ```python
            if data_type == P_TYPE:
                if data_format == FORM_PDB:
                    return FOLDSEEK, None
                elif data_format == FORM_FASTA:
                    order = [DIAMOND, MMSEQS2, CDHIT, MMSEQSPP]
                    for method in order:
                        if INSTALLED[method]:
                            return method, None
            else:
                ...
        ```

5. Tool Arguments

In step 3, we had to extract the user arguments from the `dataset` object. This is done using the `MultiYAMLParser` and
a YAML file. This YAML file must be created in the `args` directory. The YAML file must contain the following structure:

    ```yaml
        <argument name>:
          description: <description of the argument>
          type: <type of the argument, e.g., bool, int, or float>
          cardinality: <how many arguments to provide, e.g., 0 (for booleans), "?" (for one-value arguments), and "+" (for multi-value arguments)>
          default: ,<the default value of the argument>
          calls: <a list of calls, e.g., ["-a"] or ["-a", "--all"]>
          fos: <0 if this exclusively belongs to the first stage, 1 if this exclusively belongs to the second stage, and 2 if this belongs to both stages>
        ...
    ```

The fos-argument can be discarded in case there's only one stage. The YAML file must be named `diamond.yaml` (as
registered in step 2a). For tools with three or more stages, DataSAIL does no yet have a solution. Usually, not every
stage requires custom arguments (e.g., MMSEQSPP). In the example of DIAMOND, the database creation requires no user
arguments, so the YAML file only contains arguments for the `blastp` step. This step can significantly be simplified
using ChatGPT or similar tools.

Next, we need to write a checker-function for these arguments.

    ```python
        def check_diamond_arguments(args: str = "") -> Optional[Namespace]:
            """ ... """
            args = MultiYAMLParser(CDHIT).parse_args(args)
            ...
            return args
    ```

Next, we have to this checker to the `validate_user_args` function in the `cluster.py` file.

    ```python
        def validate_user_args(
                dtype: str,
                dformat: str,
                similarity: str,
                distance: str,
                tool_args: str,
        ) -> Optional[Union[Namespace, Tuple[Optional[Namespace], Optional[Namespace]]]]:
            """
            ...
            """
            sim_on, dist_on = isinstance(similarity, str), isinstance(distance, str)
            both_none = not sim_on and not dist_on
            if (sim_on and similarity.lower().startswith(CDHIT_EST)) or \
                    (both_none and get_default(dtype, dformat)[0] == CDHIT_EST):
            ...
            elif (sim_on and similarity.lower().startswith(DIAMOND)) or \
                    (both_none and get_default(dtype, dformat)[0] == DIAMOND):
                return check_diamond_arguments(tool_args)
            ...
            else:
                return None
    ```

6. Testing

Frist, we add a test to test_clustering.py. This `test_<tool_name>_<data_type>` function checks the specific (isolated)
functionality of the tool

    ```python
        @pytest.mark.nowin
        def test_diamond_protein():
            data = protein_fasta_data(DIAMOND)
            if platform.system() == "Windows":
                pytest.skip("DIAMOND is not supported on Windows")
            check_clustering(*run_diamond(data, 1, Path()), dataset=data)
    ```

Second, we add two tests to test_arg_validation.py to check if invalid arguments are detected and valid arguments are
accepted by our parser above. The tests look like this:

    ```python
        @pytest.mark.parametrize("args", [
            "--comp-based-stats 5", "--masking unknown", "--soft-masking invalid", "--evalue -0.001", "--motif-masking 2",
            ...
            "--rank-ratio -0.9", "--lambda -0.5", "--K -10"
        ])
        def test_diamond_args_checker_invalid(args):
            with pytest.raises(ValueError):
                check_diamond_arguments(args)


        @pytest.mark.parametrize("args", [
            "--comp-based-stats 2", "--masking seg", "--soft-masking tantan", "--evalue 0.001", "--motif-masking 1",
            ...
            "--K 10"
        ])
        def test_diamond_args_checker_valid(args):
            assert check_diamond_arguments(args) is not None
    ```

Lastly, we want to test the full procedure with the new clustering tool. Therefore, we have tests in
`test_custom_args.py`. For a new tool, we have to add one (or more if the tool has multiple stages) test functions.

    ```python
        def test_diamond_cargs():
            out = Path("data/pipeline/output")
            sail([
                "-o", str(out),
                "-t", "C1e",
                "-s", "0.7", "0.3",
                "--e-type", "P",
                "--e-data", str(Path('data') / 'pipeline' / 'seqs.fasta'),
                "--e-sim", "mmseqs",
                "--e-args", "--masking seg"
            ])

            assert out.is_dir()
            assert (out / "C1e").is_dir()
            assert (out / "logs").is_dir()
            assert (out / "logs" / "seqs_diamond_masking_seg.log")

            shutil.rmtree(out, ignore_errors=True)
    ```

7. Pull Request

Now, we're done have have to open a pull request on the dev branch of the DataSAIL repository. In the pull request, we
need to justify why the metric is worth including in the DataSAIL repository. This can be done by comparing results to
already existing metrics. If the new metric is better than all other metrics, we can add it to the `get_default` method
and justify that in the PR as well. Lastly, please provide links to the paper, the repo, and the documentation of the
new metric in the PR.

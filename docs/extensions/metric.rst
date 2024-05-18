##########################################################
How to Add a new Similarity or Distance Metric to DataSAIL
##########################################################

On the example of DIAMOND {links}, we demonstrate how to add a new similarity or distance metric to DataSAIL.

The DIAMOND similarity metric is a similarity metric for comparing two sets of items. It is based on the Jaccard
similarity coefficient, which is a measure of similarity between two sets. The Jaccard similarity coefficient is
defined as the size of the intersection divided by the size of the union of the two sets. The DIAMOND similarity metric
is defined as 1 minus the Jaccard similarity coefficient. The DIAMOND similarity metric is used to compare two sets of
items, such as two sets of genes, two sets of proteins, or two sets of metabolites.

0. Create a Fork of the DataSAIL Repository
###########################################

When you want to add a new feature to the DataSAIL repository, you first have to create a fork of the DataSAIL
repository. An explanation of how to do this can be found in the official GitHub documentation.

1. Installability
#################

When adding a new similarity or distance metric to DataSAIL, make sure, it is installable from Conda

2. Registration
###############

Register the new similarity or distance metric at various locations in the :code:`settings.py` file.

.. code-block:: python
    :linenos:
    :lineno-start: 69

    DIAMOND = "diamond"

and

.. code-block:: python
    :linenos:
    :lineno-start: 80

    SIM_ALGOS = [WLK, MMSEQS, MMSEQS2, MMSEQSPP, FOLDSEEK, CDHIT, CDHIT_EST, ECFP, DIAMOND, ]

and

.. code-block:: python
    :linenos:
    :lineno-start: 84

    # Check if the tools are installed
    INSTALLED = {
        # Define the check per tool
        ...
        DIAMOND: shutil.which("diamond") is not None,
        ...
    }

and

.. code-block:: python
    :linenos:
    :lineno-start: 136

    # Define the names of the YAML files
    YAML_FILE_NAMES = {
        # define the yaml file per tool
        ...
        DIAMOND: "args/diamond.yaml",
        ...
    }


3. Python Implementation
========================

Next, we create the file :code:`diamond.py` in the :code:`datasail/cluster` directory. This new file must contain a
:code:`run_<tool_name>` function, which is used to compute the similarity or distance metric. The
:code:`run_<tool_name>` function must take the following arguments:

* dataset: A DataSet-object storing the data to be clustered.
* threads: The number of threads to use for the computation.
* log_dir: An optional path to a directory where log files can be written.

The documentation of the :code:`run_<tool_name>` function must be written in the docstring of the function but can be
copied from other metric-implementations as they are very similar.

.. code-block:: python
    :linenos:
    :lineno-start: 13

    def run_diamond(dataset: DataSet, threads: int, log_dir: Optional[Path] = None) -> None:
        """
        Run Diamond on a dataset in clustering mode.

        Args:
            dataset: Dataset to be clustered.
            threads: Number of threads to be used by the clustering algorithm.
            log_dir: Directory to store the logs.
        """

Then, we have to check if the tool, we want to use is installed on the system. This can be done by using the
:code:`shutil.which` function.

.. code-block:: python
    :linenos:
    :lineno-start: 22

        if not INSTALLED[DIAMOND]:
            raise ValueError("DIAMOND is not installed.")

Next, we have to extract potential user arguments from the :code:`dataset` object. This is done using the
:code:`MultiYAMLParser` class. Here, the number of parameters to be extracted is 2, as the DIAMOND tool has two stages
from a FASTA file to the similarity matrix. If the tool has only one stage, only one set of parameters has to be
extracted (see :code:`foldseek.py`).

.. code-block:: python
    :lineno-start: 25

        parser = MultiYAMLParser(DIAMOND)
        makedb_args = parser.get_user_arguments(dataset.args, [], 0)
        blastp_args = parser.get_user_arguments(dataset.args, [], 1)

Then, we need to extract the sequences from the database into a FASTA file as only a FASTA file can be passed on to
DIAMOND.

.. code-block:: python
    :linenos:
    :lineno-start: 31

        with open("diamond.fasta", "w") as out:
            for name, seq in dataset.data.items():
                out.write(f">{name}\n{seq}\n")

Now, we construct the command line call to DIAMOND. This includes creating a folder for the tmp-files created by
DIAMOND, the actual call to DIAMOND and piping of the output to a log file or :code:`/dev/null`.

.. code-block:: python
    :linenos:
    :lineno-start: 35

        result_folder = Path("diamond_results")

        cmd = lambda x: f"mkdir {result_folder} && " \
                        f"cd {result_folder} && " \
                        f"diamond makedb --in ../diamond.fasta --db seqs.dmnd {makedb_args} {x} --threads {threads} && " \
                        f"diamond blastp --db seqs.dmnd --query ../diamond.fasta --out alis.tsv --outfmt 6 qseqid sseqid pident " \
                        f"--threads {threads} {blastp_args} {x} && " \
                        f"rm ../diamond.fasta"

        if log_dir is None:
            cmd = cmd("> /dev/null 2>&1")
        else:
            cmd = cmd(f">> {(Path(log_dir) / f'{dataset.get_name()}_mmseqspp.log').resolve()}")

        if result_folder.exists():
            cmd = f"rm -rf {result_folder} && " + cmd

Now, we're ready to run DIAMOND.

.. code-block:: python
    :linenos:
    :lineno-start: 52

        LOGGER.info("Start DIAMOND")
        LOGGER.info(cmd)
        os.system(cmd)

As something may break during the execution of the DIAMOND tool, we have to check if the output file exists.

.. code-block:: python
    :linenos:
    :lineno-start: 56

        if not (result_folder / "alis.tsv").is_file():
            raise ValueError("Something went wront with DIAMOND alignment. The output file does not exist.")

Now, it's time to harvest the results of the DIAMOND tool. This is done by reading the output file into a dataframe and
converting it into a table using :code:`df.pivot`. Then, we need to fix two details about the DIAMOND output. First,
the TSV-file has no column names, so we have to add them. Second, the similarity score is computed as pident, which is
f_ident * 100. To correct this and scale the similarities back to [0,1], we have to divide p_ident by 100.

.. code-block:: python
    :linenos:
    :lineno-start: 59

        df = pd.read_csv(result_folder / "alis.tsv", sep="\t")
        df.columns = ["query", "target", "pident"]
        df["fident"] = df["pident"] / 100
        rev = df.copy(deep=True)
        rev.columns = ["target", "query", "pident", "fident"]
        df = pd.concat([df, rev])
        df = df.groupby(["query", "target"]).agg({"fident": "mean"}).reset_index()
        table = df.pivot(index="query", columns="target", values="fident").fillna(0).to_numpy()

After deleting the temporary files, we can store the results in the :code:`dataset` object.

.. code-block:: python
    :linenos:
    :lineno-start: 68

        shutil.rmtree(result_folder)

        dataset.cluster_names = dataset.names
        dataset.cluster_map = {n: n for n in dataset.names}
        dataset.cluster_similarity = table

4. Registration -- cont'd
#########################

Now, we have to add the new similarity or distance metric to the :code:`run` function in the :code:`clustering.py` file.

.. code-block:: python
    :linenos:
    :lineno-start: 97

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

If you provide evidence in your upcoming pull request that the new metric is better than all other methods, you can add
your metric at the first place in the list in the :code:`get_default` method in the :code:`settings.py` file.

.. code-block:: python
    :linenos:
    :lineno-start: 22

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

5. Tool Arguments
#################

In step 3, we had to extract the user arguments from the :code:`dataset` object. This is done using the
:code:`MultiYAMLParser` and a YAML file. This YAML file must be created in the :code:`args` directory. The YAML file
must contain the following structure:

.. code-block:: text
    :linenos:

    <argument name>:
      description: <description of the argument>
      type: <type of the argument, e.g., bool, int, or float>
      cardinality: <how many arguments to provide, e.g., 0 (for booleans), "?" (for one-value arguments), and "+" (for multi-value arguments)>
      default: ,<the default value of the argument>
      calls: <a list of calls, e.g., ["-a"] or ["-a", "--all"]>
      fos: <0 if this exclusively belongs to the first stage, 1 if this exclusively belongs to the second stage, and 2 if this belongs to both stages>
    ...

The fos-argument can be discarded in case there's only one stage. The YAML file must be named :code:`diamond.yaml` (as
registered in step 2a). For tools with three or more stages, DataSAIL does no yet have a solution. Usually, not every
stage requires custom arguments (e.g., MMSEQSPP). In the example of DIAMOND, the database creation requires no user
arguments, so the YAML file only contains arguments for the :code:`blastp` step. This step can significantly be
simplified using ChatGPT or similar tools.

Next, we need to write a checker-function for these arguments.

.. code-block:: python
    :linenos:
    :lineno-start: 213

        def check_diamond_arguments(args: str = "") -> Optional[Namespace]:
            """ ... """
            args = MultiYAMLParser(CDHIT).parse_args(args)
            ...
            return args

Next, we have to this checker to the :code:`validate_user_args` function in the :code:`cluster.py` file.

.. code-block:: python
    :linenos:
    :lineno-start: 20

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

6. Testing
==========

First, we add a test to test_clustering.py. This :code:`test_<tool_name>_<data_type>` function in
:code:`tests/test_clustering.py` checks the specific (isolated) functionality of the tool

.. code-block:: python
    :linenos:
    :lineno-start: 230

        @pytest.mark.full
        def test_diamond_protein():
            data = protein_fasta_data(DIAMOND)
            if platform.system() == "Windows":
                pytest.skip("DIAMOND is not supported on Windows")
            run_diamond(data, 1, Path())
            check_clustering(data)

Second, we add two tests to test_arg_validation.py to check if invalid arguments are detected and valid arguments are
accepted by our parser above. The tests in :code:`tests/test_arg_validation.py` look like this:

.. code-block:: python
    :linenos:
    :lineno-start: 77

        @pytest.mark.parametrize("args", [
            "--comp-based-stats 5", "--masking unknown", "--soft-masking invalid", "--evalue -0.001",
            ...
            "--stop-match-score -0.5", "--window 0", "--ungapped-score -5", "--rank-ratio2 -0.8", "--rank-ratio -0.9",
        ])
        def test_diamond_args_checker_invalid(args):
            with pytest.raises(ValueError):
                check_diamond_arguments(args)


        @pytest.mark.parametrize("args", [
            "--comp-based-stats 2", "--masking seg", "--soft-masking tantan", "--evalue 0.001", "--motif-masking 1",
            ...
            "--rank-ratio2 0.8", "--rank-ratio 0.9", "--lambda 0.5", "--K 10"
        ])
        def test_diamond_args_checker_valid(args):
            assert check_diamond_arguments(args) is not None

Lastly, we want to test the full procedure with the new clustering tool. Therefore, we have tests in
:code:`test/test_custom_args.py`. For a new tool, we have to add one (or more if the tool has multiple stages) test
functions.

.. code-block:: python
    :linenos:
    :lineno-start: 127

        def test_diamond_cargs():
            out = Path("data/pipeline/output")
            sail([
                "-o", str(out),
                "-t", "C1e",
                "-s", "0.7", "0.3",
                "--e-type", "P",
                "--e-data", str(Path('data') / 'pipeline' / 'seqs.fasta'),
                "--e-sim", "diamond",
                "--e-args", "--gapopen 10"
            ])

            assert out.is_dir()
            assert (out / "C1e").is_dir()
            assert (out / "logs").is_dir()
            assert (out / "logs" / "seqs_diamond_gapopen_10.log")

            shutil.rmtree(out, ignore_errors=True)

7. Pull Request
###############

Now, we're done have have to open a pull request on the dev branch of the DataSAIL repository. In the pull request, we
need to justify why the metric is worth including in the DataSAIL repository. This can be done by comparing results to
already existing metrics. If the new metric is better than all other metrics, we can add it to the :code:`get_default`
method and justify that in the PR as well. Lastly, please provide links to the paper, the repo, and the documentation
of the new metric in the PR.

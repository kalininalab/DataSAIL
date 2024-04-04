########################
Contributing to DataSAIL
########################

We welcome contributions to DataSAIL. There are many ways to contribute to DataSAIL.

 1. You can report a bug using our `Bug Report Template <https://github.com/kalininalab/DataSAIL/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=>`_.
    This is the easiest way to make the developers aware of behavior that may be faulty or unexpected.
 2. Feature requests can be issued very similarly using another `Feature Request Template <https://github.com/kalininalab/DataSAIL/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.md&title=>`_.
    This is a nice way to make the developers aware of something that might be good to have, e.g., supporting new file
    formats, new tools, or new splitting algorithms.
 3. You can also file questions or ask for help in the issue tracker to get into contact with the developers and
    clarify DataSAILs functionality in points that may be interesting for other users as well. We also provide a
    `Question Template <https://github.com/kalininalab/DataSAIL/issues/new?assignees=&labels=question&projects=&template=question-help-request.md&title=>`_ for this.

A bit more involved are Pull Requests. GitHub has a nice explanation about `how to create pull requests <https://docs.github.com/de/enterprise-server@3.11/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
Following pull requests, you can directly fix bugs you found or implement features you'd like to have in DataSAIL. We
advice you to create pull requests always against the dev-branch where we accumulate new features and bug fixes for new
versions that are published in non-regular intervals. We distinguish two types of pull requests:

 4. A fast pull requests is sufficient to make small changes to DataSAIL, e.g., quickly fixing a small bug/typo/etc.
    For this, we offer a `fast RP template <https://github.com/kalininalab/DataSAIL/blob/main/.github/PULL_REQUEST_TEMPLATE/fast_pr_template.md>`_
    to collect some information that helps the developers reviewing and accepting the PR.
 5. For implementing bigger updates and new features, we request you to submit a more detailed PR following this
    `detailed PR template <https://github.com/kalininalab/DataSAIL/blob/main/.github/PULL_REQUEST_TEMPLATE/detailed_pr_template.md>`_.

Applying the PR templates to your PR can be a bit tedious, as (other then for the issue templates) GitHub does not
automatically suggest a template when creating the PR. You can either manually add the template to your PR or paste
:code:`&template=fast_pr_template.md` (or :code:`&template=detailed_pr_template.md`) in the URL to use the template.
So, the plain PR URL would look like this:

.. code-block::

    https://github.com/kalininalab/datasail/compare/dev_diamond...main?quick_pull=1

and the URL with the fast PR template would look like this:

.. code-block::

    https://github.com/kalininalab/datasail/compare/dev_diamond...main?quick_pull=1&template=fast_pr_template.md

and the URL with the detailed PR template would look like this:

.. code-block::

    https://github.com/kalininalab/datasail/compare/dev_diamond...main?quick_pull=1&template=detailed_pr_template.md

In case you have questions about contributing, the developers are happy to help you with any question during the
process.

Contributing to the Documentation
#################################

You can also contribute to the documentation of DataSAIL in several ways (for point 5 above this is desired). The
documentation is build using :code:`sphinx` and ReadTheDocs. For working on the documentation, you need to install
some more requirements listed in :code:`docs/requirements.txt`. After installing DataSAIL inside a conda environment,
simply run

.. code-block:: shell

    pip install -r docs/requirements.txt

To generate a clean build of the documentation locally, you can run

.. code-block:: shell

    rm -rf build/
    sphinx-build ./ ./build/ -a

Examples
########

DataSAIL grew to a big, complex program that may be hard to keep track of. Therefore, we are assembling examples on how
to implement certain things into DataSAIL.

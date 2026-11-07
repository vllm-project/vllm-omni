# Pinned documentation inventories

The Python inventory is copied unchanged from the official Python 3.14.0
documentation archive. It preserves standard-library API cross-references
without requiring the inventory endpoint to be available during every build.
Links still point to the public Python documentation. MkDocs strict mode remains
enabled.

- Source: <https://www.python.org/ftp/python/doc/3.14.0/python-3.14.0-docs-html.tar.bz2>
- Archive member: `python-3.14.0-docs-html/objects.inv`
- Archive SHA-256: `ddee5787a397408d895f0adaeb01c6048c067dfe9080c6753f9f92368122c4aa`
- Inventory SHA-256: `67a7bb25a9e2a1c0ca0fe7a31fef72ce47d69826c1a1a085ff87ee9d7e29f17f`
- License: [Python Software Foundation License](PYTHON-LICENSE.txt), copied from
  the archive's [licensing page](https://docs.python.org/3.14/license.html).

To update the inventory, obtain a versioned official documentation archive,
extract only `objects.inv`, verify its Sphinx header and API links, and update
the filename, `base_url`, source and hashes together.

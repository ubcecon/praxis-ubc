# SHARE-AI: "Unpacking the Black Box: Critical AI Literacies for Arts Students"

A repository for our the large TLEF project: Social science, Humanities, and Arts Research and Education: AI project.

## Important Documents

You can find the most important documents here, in our repository:

- The `CODE_OF_CONDUCT.md` file contains our project and repository code of conduct, which is expected that all members and contributors follow.

All of the other documentation is organized as outlined below.

You can find general details about this project in this document, in addition to the key files and formats.  By contributing to this project, you agree to our project code of conduct.

## Repository Guidelines

It is very important that you follow these guidelines when committing work to the repository, in order to keep things well-organized.

1.  Do _not_ commit directly to `main`.  Always commit via a merge request, from a branch.
2.  Be responsible with branches:
    * Delete old branches.
    * Don't publish un-necessary local branches.
    * Give branches sensible names.
    * Make sure you have don't commit temporary files and other materials.
3.  Use good commit messages.

### Organization

This repository is organized into several distinct parts, housed in the `main` branch's `root` directory.

* The directory `/meta` contains general repository management files.
* The directory `/project` contains all the main project files.
* The directory `/documentation` contains all the project documentation, including the style guides.

We use the [Quarto](https://quarto.org/) framework to write content and build the website, although we also can support raw `.ipynb` notebooks, as well.

### Large Files

Large files are problematic in Git: because they are stored as binaries _any_ change to them (including inconsequential ones) creates a new version of the file, which is then stored in the repository's commit history.  This means that the repo can quickly balloon in size to an unmanageable degree.  We use the [git Large File Storage](https://git-lfs.github.com/) system.

Currently, the list of filetypes which should be automatically version controlled is in the `.gitattributes` file in the main repository.  However, you should avoid committing large files whenever possible.


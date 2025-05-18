# RIGA at SMM4H-HeaRD 2025: Context-enriched classification pipeline

This is the code that used for a submission to [SMM4H-HeaRD 2025 Task 1](https://healthlanguageprocessing.org/smm4h-2025/) by RIGA team.

The development pipeline consisted of running the iterations:
1. Prepare new features;
2. Train new models;
3. Validate the feature results.

## Code structure
* data: directory is empty and used for holding the task data (provided by organizers) and DrugBank database (provided by submiting a data request for a research on the website).
* notebook: the scripts that were used for analyzing the data.
* scripts: data pre/post-processing scripts.
* train.py: the script used for training the models.

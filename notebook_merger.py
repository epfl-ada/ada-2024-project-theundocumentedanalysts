import nbformat
ABS_PATH='/Users/mehdibouchoucha/Desktop/ADA/ada-2024-project-theundocumentedanalysts/'
# Load the first notebook
with open(ABS_PATH+"cult_part_new.ipynb", "r", encoding="utf-8") as f:
    nb1 = nbformat.read(f, as_version=4)

# Load the second notebook
with open(ABS_PATH+"results_P3.ipynb", "r", encoding="utf-8") as f:
    nb2 = nbformat.read(f, as_version=4)

# Combine the cells from both notebooks
nb1.cells.extend(nb2.cells)

# Save the merged notebook
with open(ABS_PATH+"results.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb1, f)

print("Notebooks merged successfully into 'merged_notebook.ipynb'.")
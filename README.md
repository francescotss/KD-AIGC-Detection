# Knowledge Distillation for AI-Generated Image Detection: A Comparative Analysis of Continual Learning Strategies

"Abstract"

## Requirements and Usage

The code is designed to be executed on Google Colab, just open [notebook.ipynb](notebook.ipynb). All the istruction to run tranings and evaluations are inside the notebook. 

### Notebook structure

1. **Startup**: Start colab and install dependencies
2. -6. **Dataset, TL, KD, EWC, Evaluate**:  Dataset, methods and evaluation function definition
7. **Workspace**: In this section it is possible to configurate and run trainings and evaluations

### Expected dataset structure
Each dataset must be stored in
`datasets/<type>/<dataset_name>.zip`. They will be extracted at runtime in `$destination_path/<dataset_name>` (working directly on Google Drive is not recommended)


```
train/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
val/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
test/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
```

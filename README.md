[![Academic Paper](https://img.shields.io/badge/Acta_Materialia-2025-important)](https://doi.org/10.1016/j.actamat.2025.121509)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.actamat.2025.121509-royalblue)](https://doi.org/10.1016/j.actamat.2025.121509)
# DeepGlassNet: Self-Supervised Learning for Multi-Component Glass Composition Screening

## 📄 Citation 
This work is accepted for publication in [***Acta Materialia***](https://doi.org/10.1016/j.actamat.2025.121509) (a top-tier journal in materials science). Should you use this work in your research, please cite the following paper:
```
# bibtex style
@article{chen2024self,
      title={Self-Supervised Learning for Glass Composition Screening}, 
      author={Meijing Chen and Bin Liu and Ying Liu and Tianrui Li},
      journal = {Acta Materialia},
      volume = {301},
      pages = {121509},
      year = {2025},
      issn = {1359-6454},
      doi ={https://doi.org/10.1016/j.actamat.2025.121509}, 
}
```

``` 
# APA style
[1] Chen, M., Liu, B., Liu, Y., & Li, T. (2025). Self-Supervised Learning for Glass Composition Screening. Acta Materialia, 301, 121509.
```

## 1. Introduction
We present a novel self-supervised learning framework for screening multi-component glass compositions within predefined glass transition temperature (Tg) intervals (also applicable to **other multi-component material screening task**, see [**Customization Guide**](#guide) ). The composition screening task is formalized as a classification problem, aming at classifying samples that meet predifined label intervals. We introduce an innovative data augmentation strategy based on asymptotic theory to enhance training dataset robustness and improve model resilience to noise. A specialized feature extraction backbone architecture named DeepGlassNet is designed to capture complex interactions among different glass components in multi-component systems. This architecture is integrated into our self-supervised framework to optimize the Area Under Curve (AUC) classification metric. 

The framework demonstrates excellent extensibility to other multi-component material screening applications, providing an advanced methodology for efficient material design and establishing a foundation for self-supervised learning in various materials discovery tasks.

<p align='left'>
<img src='https://github.com/liubin06/DeepGlassNet/blob/main/flow.png' width='800'/>
</p>

**Figure**: Self-supervised learning workflow

The experimental dataset is derived from SciGlass Database v7.12, containing approximately 442,000 glass compositions. Each entry includes:
- Mass fractions of 18 chemical compounds
- Corresponding glass transition temperature (Tg) label

## 2. Prerequisites
- Python >= 3.7
- PyTorch 1.12.1

## 3. Code Architecture
| File | Description |
|------|-------------|
| `utils.py` | Data loading utilities and GPU-optimized dataset organization |
| `model.py` | DeepGlassNet backbone architecture implementation |
| `evaluation.py` | Model performance evaluation on validation set |
| `screening.py` | Composition screening for top-k candidate selection on test set |
| `main.py` | Central workflow controller (data processing, training, evaluation, screening) |

## 4. Configuration Flags
| Parameter | Description |
|-----------|-------------|
| `--batch_size` | Mini-batch size for training |
| `--epochs` | Maximum training epochs |
| `--learning_rate` | Optimization step size |
| `--weight_decay` | L2 regularization strength |
| `--interval` | Target Tg interval for screening |
| `--num_components` | Number of compositional features (excluding Tg label) |

## 5. Model Training
Execute the following command to initiate training:
```bash
python main.py --batch_size 1024 --epochs 100 
```


<a id="guide"></a>
## 6. Customization Guide  
This guide demonstrates how to adapt the framework for **any multi-component label screening task** (not limited to glass transition temperature, Tg).  


#### 6.1 **Data Formatting**  
(1) Structure your dataset into a single CSV file following the universal input-output format:  
- **Input features**: The **first `n` columns** must contain component or feature values (e.g., chemical compositions, material parameters). These columns collectively represent the input characteristics of the samples.
- **Target label**: The **last column** (immediately following the n input feature columns) should contain the continuous label (e.g., glass transition temperature for glassy materials, yield strength for alloy systems).
- Clarification: All data (both input features and target label) are consolidated into one CSV file with a strict column order:  
`[Feature Column 1], [Feature Column 2], ..., [Feature Column n], [Target Label Column]`  

(2) **Dataset split**:  Split the data into train/validation set.
  - Training set: Save as `train.csv` (contains both features and labels for model training).  
  - Validation set: Save as `validation.csv` (contains both features and labels for model performance evaluation and hyperparameter fine-tuning).  

(3) **Prepare YOUR Screening set**:
  - First, generate potential component combinations. This can be achieved via methods such as enumeration or theoretical derivation; these combinations should represent **theoretically feasible, potential** unseen compositions without sample labels. In general, a larger sample size is preferable to ensure comprehensive coverage of candidate compositions.
  - Save the generated component combinations as `test.csv`. Critically, this file must contain **only the `n` component/feature columns** (i.e., no label column). This file will serve as the input for screening the most promising candidate samples from the screening set.
  - The model will screen and rank the top-k most promising samples from these potential compositions, thereby effectively narrowing the sample search space for subsequent experimental design and preparation.


#### 6.2 **Define Your Target Label Interval**  
Specify the continuous label interval for screening in `main.py`. This can be any numerical range relevant to your task (e.g., strength thresholds, temperature ranges, etc.):  
```python  
# In main.py  
interval = [LOWER_BOUND, UPPER_BOUND]  # Replace with your target label interval (e.g., [200, 300] for a strength metric)  
```  


#### 6.3 **Configure Feature Dimensions**  
Set the number of input features (`n`) to match your dataset’s component count.   
```python  
# In main.py  
parser.add_argument('--num_components', type=int, default=NUM_FEATURES)  # Replace "NUM_FEATURES" with your actual feature count (e.g., 5 for a 5-component material)  
```  


#### 6.4 **Execute the Screening Pipeline**  
Run the following command to train the model and generate top candidates that fall within your specified label interval. The framework automatically adapts to your task’s feature-label mapping:  
```bash  
python main.py  
# Output: Top-10 candidate samples from `test.csv` whose predicted labels match your interval.  
```  


#### 6.5 **Generalization Notes**  
- **Task flexibility**: The framework is applicable to **other multi-component material screening task**  .  
- **Physical constraints**: Ensure input features comply with domain rules.  


#### 6.6 **Further Assistance**  
For task-specific adjustments or technical support, contact **Bin Liu**: binliu@swjtu.edu.cn
   
## 7. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



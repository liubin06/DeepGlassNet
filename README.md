
# DeepGlassNet: Self-Supervised Learning for Multi-Component Glass Composition Screening

## 1. Introduction
We present a novel self-supervised learning framework for screening multi-component glass compositions within predefined glass transition temperature (Tg) intervals (also applicable to **other multi-component label screening task**, see [**Customization Guide**](#guide) ). The composition screening task is formalized as a classification problem, aming at classifying samples that meet predifined label intervals. We introduce an innovative data augmentation strategy based on asymptotic theory to enhance training dataset robustness and improve model resilience to noise. A specialized feature extraction backbone architecture named DeepGlassNet is designed to capture complex interactions among different glass components in multi-component systems. This architecture is integrated into our self-supervised framework to optimize the Area Under Curve (AUC) classification metric. 

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
Organize your dataset to fit the universal input-output structure:  
- **Input features**: `n` columns for component/feature values (e.g., chemical compositions, material parameters).  
- **Target label**: A single column for the continuous label to screen (e.g., Tg for glass, yield strength for alloys, etc.), placed as the last column.  
- **Dataset split**:  
  - Training set: Save as `train.csv` (contains both features and labels).  
  - Validation set: Save as `validation.csv` (contains both features and labels).  
  - Screening set: Save as `test.csv` (contains **only** the `n` component/feature columns, **no label**), used for screen most promissing candidate samples.  


#### 6.2 **Define Your Target Label Interval**  
Specify the continuous label interval for screening in `main.py`. This can be any numerical range relevant to your task (e.g., strength thresholds, temperature ranges, etc.):  
```python  
# In main.py  
interval = [LOWER_BOUND, UPPER_BOUND]  # Replace with your target label interval (e.g., [200, 300] for a strength metric)  
```  


#### 6.3 **Configure Feature Dimensions**  
Set the number of input features (`n`) to match your dataset’s component count. This parameter is **task-agnostic** and applies to any multi-component scenario:  
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
- **Task flexibility**: The framework is designed for **any continuous label screening task** (e.g., material property optimization, chemical reaction yield prediction, sensor signal threshold detection).  
- **Physical constraints**: For material-specific tasks, ensure input features comply with domain rules (e.g., component ratios summing to 100%).  


#### 6.6 **Further Assistance**  
For task-specific adjustments or technical support, contact `binliu@swjtu.edu.cn`. 
   
## 7. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 8. Citation
If using this work in your research, please consider citing the following paper:
```bibtex
@article{chen2024self,
      title={Self-Supervised Learning for Glass Composition Screening}, 
      author={Meijing Chen and Bin Liu and Ying Liu and Tianrui Li},
      year={2024},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.24083v2}, 
}
```

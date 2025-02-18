
# DeepGlassNet: Self-Supervised Learning for Multi-Component Glass Composition Screening

## 1. Introduction
We present a novel self-supervised learning framework for screening multi-component glass compositions within predefined glass transition temperature (Tg) intervals. The composition screening task is formalized as a classification problem, where we introduce an innovative data augmentation strategy based on asymptotic theory to enhance training dataset robustness and improve model resilience to noise. A specialized feature extraction backbone architecture named DeepGlassNet is designed to capture complex interactions among different glass components in multi-component systems. This architecture is integrated into our self-supervised framework to optimize the Area Under Curve (AUC) classification metric. 

The framework demonstrates excellent extensibility to other multi-component material screening applications, providing an advanced methodology for efficient glass design and establishing a foundation for self-supervised learning in various materials discovery tasks.

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
| `screen.py` | Composition screening for top-k candidate selection on test set |
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

## 6. Customization Guide
To train with proprietary datasets:

1. **Data Formatting**:
   - Arrange data with `n` component columns followed by Tg label at last column
   - Split data as `tran_tg.csv` and `validation_tg.csv`.
   - Replace the data you want to scan with the `test_tg.csv` file in the same format.

2. **Configuration**:
   ```python
   # In main.py
   parser.add_argument('--num_components', type=int, default=SET_YOUR_COMPONENT_NUM)  # Set number of components
   interval = [300, 400]  # Set YOUR desired screening temperature range (℃)
   ```

3. **Execution**:
   ```bash
   python main.py 
   ```

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

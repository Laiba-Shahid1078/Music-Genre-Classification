# Model Experiment Results

| Model | Main Change | Parameters | Test Accuracy | Test Loss | Outcome |
|---|---|---:|---:|---:|---|
| V1 - Baseline CNN | Basic CNN | 7,393,034 | 64% | 1.324 | Strong baseline; overfitting observed |
| V2 - GAP | Flatten replaced with GlobalAveragePooling2D | 28,426 | 32% | 1.942 | Underfitting |
| V3 - Deeper CNN + BatchNorm | Added third Conv2D block and BatchNormalization | 3,306,250 | 10% | 22.070 | Unstable/poor learning |
| V4 - Lower Learning Rate | LR 0.001 → 0.0001 + callbacks | 7,393,034 | 51% | 1.482 | More controlled but worse performance |
| Final Model | V1 architecture + EarlyStopping + ReduceLROnPlateau | 7,393,034 | 64% | 1.071 | Selected for deployment |

## Final Model

- Test Accuracy: 64%
- Test Loss: 1.071
- Best Validation Accuracy: 68%
- Best Validation Loss: 0.9745
- Selected Weights: Epoch 15

## Key Learning

The experiments showed that model complexity and learning rate both significantly affected generalization. The simpler GAP model underfit, while the deeper CNN experiment became unstable. The final model retained the baseline architecture and used training callbacks to control optimization.

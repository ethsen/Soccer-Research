# Test Notebooks – Model Inference & Visualization

This folder contains notebooks used **only for qualitative evaluation** of trained models. No training or preprocessing is performed here.

## Purpose
The test notebooks demonstrate how to:
- Sample a single example from the **validation set**
- Run **model inference** on that sample
- **Visualize** the resulting probability maps on the pitch

## Workflow
1. Load the validation dataset  
2. Sample a validation example  
3. Run the trained model in `eval()` mode  
4. Visualize the model outputs (destination and/or completion maps)

No additional steps are required.

## Notes
- All examples are drawn from the **validation split**
- Models are used strictly for inference
- Only path variables may need adjustment for your environment

Running the notebook end-to-end will produce the visualizations.


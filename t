2025-05-11 07:42:19,371 [INFO] Using device: cuda
2025-05-11 07:42:19,372 [INFO] Using 4 workers for DataLoaders
2025-05-11 07:42:19,379 [INFO] --- Starting Fold 1/5 ---
2025-05-11 07:42:19,383 [INFO] Initializing VGG11 model.
Downloading: "https://download.pytorch.org/models/vgg11-8a719046.pth" to /home/marvin/.cache/torch/hub/checkpoints/vgg11-8a719046.pth
100%|██████████| 507M/507M [00:50<00:00, 10.5MB/s] 
2025-05-11 07:43:11,885 [INFO] Adjusted VGG11's existing dropout rates at classifier indices [2, 5] to 0.5
2025-05-11 07:43:11,885 [INFO] Replaced final layer of VGG11 (classifier[6]) for 2 output classes.
2025-05-11 07:43:12,457 [INFO] --- Phase 1: Training Head ---
Training Epoch (Head):   0%|          | 0/1435 [00:00<?, ?it/s]/home/marvin/Developer/NoSmiles/local/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:125: UserWarning: Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:310.)
  return F.linear(input, self.weight, self.bias)
2025-05-11 07:45:12,630 [INFO] Fold 1 Phase 1 - Epoch 1/7, Train Loss: 0.6902, Time: 120.17s
2025-05-11 07:47:12,936 [INFO] Fold 1 Phase 1 - Epoch 2/7, Train Loss: 0.5248, Time: 120.31s
2025-05-11 07:49:13,254 [INFO] Fold 1 Phase 1 - Epoch 3/7, Train Loss: 0.6102, Time: 120.32s
2025-05-11 07:51:13,218 [INFO] Fold 1 Phase 1 - Epoch 4/7, Train Loss: 0.6189, Time: 119.96s
2025-05-11 07:53:13,399 [INFO] Fold 1 Phase 1 - Epoch 5/7, Train Loss: 0.4765, Time: 120.18s
2025-05-11 07:55:13,285 [INFO] Fold 1 Phase 1 - Epoch 6/7, Train Loss: 0.4791, Time: 119.89s
2025-05-11 07:57:12,084 [INFO] Fold 1 Phase 1 - Epoch 7/7, Train Loss: 0.4417, Time: 118.80s
2025-05-11 07:57:12,085 [INFO] --- Phase 2: Fine-tuning Full Model ---
2025-05-11 08:00:49,216 [INFO] Fold 1 Phase 2 - Epoch 1/20 (Total: 8), LR: [1e-05, 0.001], Train Loss: 0.5368, Val Loss: 0.4020, Val Acc: 0.8664, Val F1: 0.7283, Val AUC: 0.8929, Time: 217.13s
2025-05-11 08:00:50,837 [INFO]   -> New best F1: 0.7283 at epoch 8. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:04:26,932 [INFO] Fold 1 Phase 2 - Epoch 2/20 (Total: 9), LR: [1e-05, 0.001], Train Loss: 0.4289, Val Loss: 0.3033, Val Acc: 0.8891, Val F1: 0.7732, Val AUC: 0.9239, Time: 216.09s
2025-05-11 08:04:31,511 [INFO]   -> New best F1: 0.7732 at epoch 9. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:08:07,799 [INFO] Fold 1 Phase 2 - Epoch 3/20 (Total: 10), LR: [1e-05, 0.001], Train Loss: 0.3596, Val Loss: 0.3303, Val Acc: 0.8800, Val F1: 0.7801, Val AUC: 0.9346, Time: 216.29s
2025-05-11 08:08:12,341 [INFO]   -> New best F1: 0.7801 at epoch 10. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:11:48,905 [INFO] Fold 1 Phase 2 - Epoch 4/20 (Total: 11), LR: [1e-05, 0.001], Train Loss: 0.3227, Val Loss: 0.2685, Val Acc: 0.8925, Val F1: 0.7933, Val AUC: 0.9439, Time: 216.56s
2025-05-11 08:11:53,466 [INFO]   -> New best F1: 0.7933 at epoch 11. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:15:29,834 [INFO] Fold 1 Phase 2 - Epoch 5/20 (Total: 12), LR: [1e-05, 0.001], Train Loss: 0.3072, Val Loss: 0.2562, Val Acc: 0.9079, Val F1: 0.8085, Val AUC: 0.9421, Time: 216.37s
2025-05-11 08:15:34,344 [INFO]   -> New best F1: 0.8085 at epoch 12. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:19:10,653 [INFO] Fold 1 Phase 2 - Epoch 6/20 (Total: 13), LR: [1e-05, 0.001], Train Loss: 0.2844, Val Loss: 0.3056, Val Acc: 0.9056, Val F1: 0.7955, Val AUC: 0.9334, Time: 216.31s
2025-05-11 08:19:10,654 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 08:22:46,896 [INFO] Fold 1 Phase 2 - Epoch 7/20 (Total: 14), LR: [1e-05, 0.001], Train Loss: 0.2706, Val Loss: 0.2421, Val Acc: 0.9131, Val F1: 0.8174, Val AUC: 0.9509, Time: 216.24s
2025-05-11 08:22:51,490 [INFO]   -> New best F1: 0.8174 at epoch 14. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:26:27,694 [INFO] Fold 1 Phase 2 - Epoch 8/20 (Total: 15), LR: [1e-05, 0.001], Train Loss: 0.2578, Val Loss: 0.2616, Val Acc: 0.9127, Val F1: 0.8104, Val AUC: 0.9438, Time: 216.20s
2025-05-11 08:26:27,696 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 08:30:03,854 [INFO] Fold 1 Phase 2 - Epoch 9/20 (Total: 16), LR: [1e-05, 0.001], Train Loss: 0.2524, Val Loss: 0.2688, Val Acc: 0.9033, Val F1: 0.8146, Val AUC: 0.9484, Time: 216.16s
2025-05-11 08:30:03,855 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 08:33:39,916 [INFO] Fold 1 Phase 2 - Epoch 10/20 (Total: 17), LR: [1e-05, 0.001], Train Loss: 0.2389, Val Loss: 0.2578, Val Acc: 0.9169, Val F1: 0.8279, Val AUC: 0.9486, Time: 216.06s
2025-05-11 08:33:44,469 [INFO]   -> New best F1: 0.8279 at epoch 17. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:37:20,785 [INFO] Fold 1 Phase 2 - Epoch 11/20 (Total: 18), LR: [1e-05, 0.001], Train Loss: 0.2315, Val Loss: 0.3380, Val Acc: 0.9117, Val F1: 0.8176, Val AUC: 0.9415, Time: 216.32s
2025-05-11 08:37:20,787 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 08:40:57,111 [INFO] Fold 1 Phase 2 - Epoch 12/20 (Total: 19), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1934, Val Loss: 0.2791, Val Acc: 0.9105, Val F1: 0.8245, Val AUC: 0.9514, Time: 216.32s
2025-05-11 08:40:57,112 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 08:44:33,747 [INFO] Fold 1 Phase 2 - Epoch 13/20 (Total: 20), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1834, Val Loss: 0.2810, Val Acc: 0.9155, Val F1: 0.8310, Val AUC: 0.9526, Time: 216.63s
2025-05-11 08:44:38,323 [INFO]   -> New best F1: 0.8310 at epoch 20. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 08:48:14,666 [INFO] Fold 1 Phase 2 - Epoch 14/20 (Total: 21), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1802, Val Loss: 0.2814, Val Acc: 0.9127, Val F1: 0.8279, Val AUC: 0.9524, Time: 216.34s
2025-05-11 08:48:14,668 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 08:51:50,851 [INFO] Fold 1 Phase 2 - Epoch 15/20 (Total: 22), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1753, Val Loss: 0.2683, Val Acc: 0.9148, Val F1: 0.8304, Val AUC: 0.9523, Time: 216.18s
2025-05-11 08:51:50,853 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 08:55:27,152 [INFO] Fold 1 Phase 2 - Epoch 16/20 (Total: 23), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1688, Val Loss: 0.2743, Val Acc: 0.9127, Val F1: 0.8275, Val AUC: 0.9525, Time: 216.30s
2025-05-11 08:55:27,154 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 08:59:03,363 [INFO] Fold 1 Phase 2 - Epoch 17/20 (Total: 24), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1729, Val Loss: 0.2732, Val Acc: 0.9150, Val F1: 0.8310, Val AUC: 0.9526, Time: 216.21s
2025-05-11 08:59:07,829 [INFO]   -> New best F1: 0.8310 at epoch 24. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 09:02:44,404 [INFO] Fold 1 Phase 2 - Epoch 18/20 (Total: 25), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1710, Val Loss: 0.2727, Val Acc: 0.9150, Val F1: 0.8310, Val AUC: 0.9525, Time: 216.57s
2025-05-11 09:02:44,406 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 09:06:20,669 [INFO] Fold 1 Phase 2 - Epoch 19/20 (Total: 26), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1649, Val Loss: 0.2757, Val Acc: 0.9154, Val F1: 0.8318, Val AUC: 0.9523, Time: 216.26s
2025-05-11 09:06:25,212 [INFO]   -> New best F1: 0.8318 at epoch 26. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 09:10:01,826 [INFO] Fold 1 Phase 2 - Epoch 20/20 (Total: 27), LR: [1.0000000000000004e-08, 1.0000000000000002e-06], Train Loss: 0.1702, Val Loss: 0.2751, Val Acc: 0.9155, Val F1: 0.8321, Val AUC: 0.9523, Time: 216.61s
2025-05-11 09:10:06,399 [INFO]   -> New best F1: 0.8321 at epoch 27. Checkpoint saved to models_checkpointed/best_model_fold_1.pth
2025-05-11 09:10:06,400 [INFO] Loading best model from models_checkpointed/best_model_fold_1.pth (Epoch 27, F1: 0.8321)
2025-05-11 09:10:07,305 [INFO] --- Evaluating Best Model for Fold 1 ---
2025-05-11 09:10:22,585 [INFO] Fold 1 Final Validation Results (Best Model):
2025-05-11 09:10:22,587 [INFO]   Accuracy:  0.9155
2025-05-11 09:10:22,588 [INFO]   F1 Score:  0.8321
2025-05-11 09:10:22,590 [INFO]   Precision: 0.8313
2025-05-11 09:10:22,592 [INFO]   Recall:    0.8330
2025-05-11 09:10:22,594 [INFO]   AUC:       0.9523
2025-05-11 09:10:22,596 [INFO]   Loss:      0.2751
2025-05-11 09:10:22,598 [INFO]   Confusion Matrix:
2025-05-11 09:10:22,601 [INFO] 
[[4055  244]
 [ 241 1202]]
2025-05-11 09:10:22,603 [INFO] --- Fold 1 completed in 5283.22s ---
2025-05-11 09:10:22,605 [INFO] --- Starting Fold 2/5 ---
2025-05-11 09:10:22,610 [INFO] Initializing VGG11 model.
2025-05-11 09:10:23,671 [INFO] Adjusted VGG11's existing dropout rates at classifier indices [2, 5] to 0.5
2025-05-11 09:10:23,678 [INFO] Replaced final layer of VGG11 (classifier[6]) for 2 output classes.
2025-05-11 09:10:23,801 [INFO] --- Phase 1: Training Head ---
2025-05-11 09:12:24,808 [INFO] Fold 2 Phase 1 - Epoch 1/7, Train Loss: 0.7011, Time: 121.00s
2025-05-11 09:14:25,569 [INFO] Fold 2 Phase 1 - Epoch 2/7, Train Loss: 0.5257, Time: 120.76s
2025-05-11 09:16:26,566 [INFO] Fold 2 Phase 1 - Epoch 3/7, Train Loss: 0.5438, Time: 121.00s
2025-05-11 09:18:27,617 [INFO] Fold 2 Phase 1 - Epoch 4/7, Train Loss: 0.5205, Time: 121.05s
2025-05-11 09:20:28,735 [INFO] Fold 2 Phase 1 - Epoch 5/7, Train Loss: 0.5568, Time: 121.12s
2025-05-11 09:22:29,716 [INFO] Fold 2 Phase 1 - Epoch 6/7, Train Loss: 0.5555, Time: 120.98s
2025-05-11 09:24:30,949 [INFO] Fold 2 Phase 1 - Epoch 7/7, Train Loss: 0.5442, Time: 121.23s
2025-05-11 09:24:30,951 [INFO] --- Phase 2: Fine-tuning Full Model ---
2025-05-11 09:28:07,782 [INFO] Fold 2 Phase 2 - Epoch 1/20 (Total: 8), LR: [1e-05, 0.001], Train Loss: 0.5695, Val Loss: 0.4093, Val Acc: 0.8391, Val F1: 0.7139, Val AUC: 0.8897, Time: 216.83s
2025-05-11 09:28:09,507 [INFO]   -> New best F1: 0.7139 at epoch 8. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 09:31:54,631 [INFO] Fold 2 Phase 2 - Epoch 2/20 (Total: 9), LR: [1e-05, 0.001], Train Loss: 0.4370, Val Loss: 0.3024, Val Acc: 0.8922, Val F1: 0.7850, Val AUC: 0.9283, Time: 225.12s
2025-05-11 09:31:59,244 [INFO]   -> New best F1: 0.7850 at epoch 9. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 09:35:57,046 [INFO] Fold 2 Phase 2 - Epoch 3/20 (Total: 10), LR: [1e-05, 0.001], Train Loss: 0.3747, Val Loss: 0.2776, Val Acc: 0.9035, Val F1: 0.7839, Val AUC: 0.9423, Time: 237.80s
2025-05-11 09:35:57,054 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 09:39:55,819 [INFO] Fold 2 Phase 2 - Epoch 4/20 (Total: 11), LR: [1e-05, 0.001], Train Loss: 0.3377, Val Loss: 0.2724, Val Acc: 0.9058, Val F1: 0.8113, Val AUC: 0.9400, Time: 238.76s
2025-05-11 09:40:00,407 [INFO]   -> New best F1: 0.8113 at epoch 11. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 09:43:59,139 [INFO] Fold 2 Phase 2 - Epoch 5/20 (Total: 12), LR: [1e-05, 0.001], Train Loss: 0.3115, Val Loss: 0.2647, Val Acc: 0.9107, Val F1: 0.8224, Val AUC: 0.9471, Time: 238.72s
2025-05-11 09:44:03,698 [INFO]   -> New best F1: 0.8224 at epoch 12. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 09:48:04,159 [INFO] Fold 2 Phase 2 - Epoch 6/20 (Total: 13), LR: [1e-05, 0.001], Train Loss: 0.3022, Val Loss: 0.2521, Val Acc: 0.9086, Val F1: 0.8216, Val AUC: 0.9506, Time: 240.45s
2025-05-11 09:48:04,169 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 09:52:04,770 [INFO] Fold 2 Phase 2 - Epoch 7/20 (Total: 14), LR: [1e-05, 0.001], Train Loss: 0.2831, Val Loss: 0.2467, Val Acc: 0.9178, Val F1: 0.8375, Val AUC: 0.9513, Time: 240.59s
2025-05-11 09:52:09,413 [INFO]   -> New best F1: 0.8375 at epoch 14. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 09:56:10,249 [INFO] Fold 2 Phase 2 - Epoch 8/20 (Total: 15), LR: [1e-05, 0.001], Train Loss: 0.2843, Val Loss: 0.2493, Val Acc: 0.9122, Val F1: 0.8196, Val AUC: 0.9487, Time: 240.83s
2025-05-11 09:56:10,259 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 10:00:10,764 [INFO] Fold 2 Phase 2 - Epoch 9/20 (Total: 16), LR: [1e-05, 0.001], Train Loss: 0.2603, Val Loss: 0.2526, Val Acc: 0.9101, Val F1: 0.8272, Val AUC: 0.9553, Time: 240.49s
2025-05-11 10:00:10,769 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 10:04:11,633 [INFO] Fold 2 Phase 2 - Epoch 10/20 (Total: 17), LR: [1e-05, 0.001], Train Loss: 0.2527, Val Loss: 0.2410, Val Acc: 0.9169, Val F1: 0.8334, Val AUC: 0.9547, Time: 240.85s
2025-05-11 10:04:11,643 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 10:08:13,052 [INFO] Fold 2 Phase 2 - Epoch 11/20 (Total: 18), LR: [1e-05, 0.001], Train Loss: 0.2431, Val Loss: 0.2886, Val Acc: 0.8985, Val F1: 0.8171, Val AUC: 0.9559, Time: 241.40s
2025-05-11 10:08:13,063 [INFO]   -> F1 did not improve. Patience: 4/7
2025-05-11 10:12:15,130 [INFO] Fold 2 Phase 2 - Epoch 12/20 (Total: 19), LR: [1e-05, 0.001], Train Loss: 0.2291, Val Loss: 0.2917, Val Acc: 0.9150, Val F1: 0.8348, Val AUC: 0.9538, Time: 242.06s
2025-05-11 10:12:15,141 [INFO]   -> F1 did not improve. Patience: 5/7
2025-05-11 10:16:17,639 [INFO] Fold 2 Phase 2 - Epoch 13/20 (Total: 20), LR: [1e-05, 0.001], Train Loss: 0.2231, Val Loss: 0.2442, Val Acc: 0.9185, Val F1: 0.8398, Val AUC: 0.9551, Time: 242.49s
2025-05-11 10:16:22,219 [INFO]   -> New best F1: 0.8398 at epoch 20. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 10:20:24,492 [INFO] Fold 2 Phase 2 - Epoch 14/20 (Total: 21), LR: [1e-05, 0.001], Train Loss: 0.2118, Val Loss: 0.3075, Val Acc: 0.9204, Val F1: 0.8397, Val AUC: 0.9552, Time: 242.26s
2025-05-11 10:20:24,503 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 10:24:27,601 [INFO] Fold 2 Phase 2 - Epoch 15/20 (Total: 22), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1813, Val Loss: 0.2577, Val Acc: 0.9244, Val F1: 0.8499, Val AUC: 0.9578, Time: 243.09s
2025-05-11 10:24:32,233 [INFO]   -> New best F1: 0.8499 at epoch 22. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 10:28:36,132 [INFO] Fold 2 Phase 2 - Epoch 16/20 (Total: 23), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1718, Val Loss: 0.2561, Val Acc: 0.9255, Val F1: 0.8511, Val AUC: 0.9586, Time: 243.89s
2025-05-11 10:28:40,913 [INFO]   -> New best F1: 0.8511 at epoch 23. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 10:32:44,392 [INFO] Fold 2 Phase 2 - Epoch 17/20 (Total: 24), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1696, Val Loss: 0.2648, Val Acc: 0.9251, Val F1: 0.8528, Val AUC: 0.9594, Time: 243.47s
2025-05-11 10:32:48,971 [INFO]   -> New best F1: 0.8528 at epoch 24. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 10:36:53,816 [INFO] Fold 2 Phase 2 - Epoch 18/20 (Total: 25), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1666, Val Loss: 0.2567, Val Acc: 0.9232, Val F1: 0.8495, Val AUC: 0.9588, Time: 244.83s
2025-05-11 10:36:53,828 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 10:40:59,130 [INFO] Fold 2 Phase 2 - Epoch 19/20 (Total: 26), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1564, Val Loss: 0.2617, Val Acc: 0.9246, Val F1: 0.8517, Val AUC: 0.9588, Time: 245.30s
2025-05-11 10:40:59,143 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 10:45:04,880 [INFO] Fold 2 Phase 2 - Epoch 20/20 (Total: 27), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1542, Val Loss: 0.2633, Val Acc: 0.9258, Val F1: 0.8533, Val AUC: 0.9585, Time: 245.73s
2025-05-11 10:45:09,517 [INFO]   -> New best F1: 0.8533 at epoch 27. Checkpoint saved to models_checkpointed/best_model_fold_2.pth
2025-05-11 10:45:09,523 [INFO] Loading best model from models_checkpointed/best_model_fold_2.pth (Epoch 27, F1: 0.8533)
2025-05-11 10:45:10,426 [INFO] --- Evaluating Best Model for Fold 2 ---
2025-05-11 10:45:26,777 [INFO] Fold 2 Final Validation Results (Best Model):
2025-05-11 10:45:26,782 [INFO]   Accuracy:  0.9258
2025-05-11 10:45:26,795 [INFO]   F1 Score:  0.8533
2025-05-11 10:45:26,808 [INFO]   Precision: 0.8480
2025-05-11 10:45:26,814 [INFO]   Recall:    0.8586
2025-05-11 10:45:26,827 [INFO]   AUC:       0.9585
2025-05-11 10:45:26,840 [INFO]   Loss:      0.2633
2025-05-11 10:45:26,846 [INFO]   Confusion Matrix:
2025-05-11 10:45:26,859 [INFO] 
[[4077  222]
 [ 204 1239]]
2025-05-11 10:45:26,872 [INFO] --- Fold 2 completed in 5704.27s ---
2025-05-11 10:45:26,878 [INFO] --- Starting Fold 3/5 ---
2025-05-11 10:45:26,893 [INFO] Initializing VGG11 model.
2025-05-11 10:45:27,960 [INFO] Adjusted VGG11's existing dropout rates at classifier indices [2, 5] to 0.5
2025-05-11 10:45:27,963 [INFO] Replaced final layer of VGG11 (classifier[6]) for 2 output classes.
2025-05-11 10:45:28,095 [INFO] --- Phase 1: Training Head ---
2025-05-11 10:47:48,651 [INFO] Fold 3 Phase 1 - Epoch 1/7, Train Loss: 0.6801, Time: 140.55s
2025-05-11 10:50:11,099 [INFO] Fold 3 Phase 1 - Epoch 2/7, Train Loss: 0.5264, Time: 142.44s
2025-05-11 10:52:34,256 [INFO] Fold 3 Phase 1 - Epoch 3/7, Train Loss: 0.6765, Time: 143.14s
2025-05-11 10:54:58,713 [INFO] Fold 3 Phase 1 - Epoch 4/7, Train Loss: 0.5104, Time: 144.45s
2025-05-11 10:57:25,176 [INFO] Fold 3 Phase 1 - Epoch 5/7, Train Loss: 0.5179, Time: 146.45s
2025-05-11 10:59:53,230 [INFO] Fold 3 Phase 1 - Epoch 6/7, Train Loss: 0.5564, Time: 148.04s
2025-05-11 11:02:22,604 [INFO] Fold 3 Phase 1 - Epoch 7/7, Train Loss: 0.6089, Time: 149.36s
2025-05-11 11:02:22,618 [INFO] --- Phase 2: Fine-tuning Full Model ---
2025-05-11 11:06:31,457 [INFO] Fold 3 Phase 2 - Epoch 1/20 (Total: 8), LR: [1e-05, 0.001], Train Loss: 0.5433, Val Loss: 0.4035, Val Acc: 0.8582, Val F1: 0.6658, Val AUC: 0.8855, Time: 248.82s
2025-05-11 11:06:33,151 [INFO]   -> New best F1: 0.6658 at epoch 8. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:10:41,685 [INFO] Fold 3 Phase 2 - Epoch 2/20 (Total: 9), LR: [1e-05, 0.001], Train Loss: 0.4577, Val Loss: 0.3177, Val Acc: 0.8819, Val F1: 0.7506, Val AUC: 0.9120, Time: 248.52s
2025-05-11 11:10:46,348 [INFO]   -> New best F1: 0.7506 at epoch 9. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:14:55,365 [INFO] Fold 3 Phase 2 - Epoch 3/20 (Total: 10), LR: [1e-05, 0.001], Train Loss: 0.3727, Val Loss: 0.3032, Val Acc: 0.8877, Val F1: 0.7795, Val AUC: 0.9351, Time: 249.01s
2025-05-11 11:15:00,018 [INFO]   -> New best F1: 0.7795 at epoch 10. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:19:09,380 [INFO] Fold 3 Phase 2 - Epoch 4/20 (Total: 11), LR: [1e-05, 0.001], Train Loss: 0.3317, Val Loss: 0.2744, Val Acc: 0.8988, Val F1: 0.7943, Val AUC: 0.9378, Time: 249.35s
2025-05-11 11:19:14,029 [INFO]   -> New best F1: 0.7943 at epoch 11. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:23:24,580 [INFO] Fold 3 Phase 2 - Epoch 5/20 (Total: 12), LR: [1e-05, 0.001], Train Loss: 0.3100, Val Loss: 0.3144, Val Acc: 0.8939, Val F1: 0.7940, Val AUC: 0.9288, Time: 250.54s
2025-05-11 11:23:24,595 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 11:27:35,756 [INFO] Fold 3 Phase 2 - Epoch 6/20 (Total: 13), LR: [1e-05, 0.001], Train Loss: 0.2969, Val Loss: 0.2543, Val Acc: 0.9108, Val F1: 0.8190, Val AUC: 0.9474, Time: 251.15s
2025-05-11 11:27:40,270 [INFO]   -> New best F1: 0.8190 at epoch 13. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:31:52,023 [INFO] Fold 3 Phase 2 - Epoch 7/20 (Total: 14), LR: [1e-05, 0.001], Train Loss: 0.2825, Val Loss: 0.2844, Val Acc: 0.8969, Val F1: 0.8011, Val AUC: 0.9411, Time: 251.75s
2025-05-11 11:31:52,030 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 11:36:04,770 [INFO] Fold 3 Phase 2 - Epoch 8/20 (Total: 15), LR: [1e-05, 0.001], Train Loss: 0.2646, Val Loss: 0.2760, Val Acc: 0.8981, Val F1: 0.8067, Val AUC: 0.9470, Time: 252.72s
2025-05-11 11:36:04,777 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 11:40:18,269 [INFO] Fold 3 Phase 2 - Epoch 9/20 (Total: 16), LR: [1e-05, 0.001], Train Loss: 0.2565, Val Loss: 0.2689, Val Acc: 0.9098, Val F1: 0.8217, Val AUC: 0.9445, Time: 253.48s
2025-05-11 11:40:22,884 [INFO]   -> New best F1: 0.8217 at epoch 16. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:44:43,177 [INFO] Fold 3 Phase 2 - Epoch 10/20 (Total: 17), LR: [1e-05, 0.001], Train Loss: 0.2551, Val Loss: 0.2574, Val Acc: 0.9145, Val F1: 0.8201, Val AUC: 0.9464, Time: 260.28s
2025-05-11 11:44:43,195 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 11:49:13,613 [INFO] Fold 3 Phase 2 - Epoch 11/20 (Total: 18), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.2124, Val Loss: 0.2672, Val Acc: 0.9141, Val F1: 0.8312, Val AUC: 0.9512, Time: 270.40s
2025-05-11 11:49:18,278 [INFO]   -> New best F1: 0.8312 at epoch 18. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:53:49,404 [INFO] Fold 3 Phase 2 - Epoch 12/20 (Total: 19), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.2021, Val Loss: 0.2588, Val Acc: 0.9187, Val F1: 0.8363, Val AUC: 0.9502, Time: 271.11s
2025-05-11 11:53:54,158 [INFO]   -> New best F1: 0.8363 at epoch 19. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 11:58:25,646 [INFO] Fold 3 Phase 2 - Epoch 13/20 (Total: 20), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1978, Val Loss: 0.2545, Val Acc: 0.9178, Val F1: 0.8374, Val AUC: 0.9511, Time: 271.47s
2025-05-11 11:58:30,297 [INFO]   -> New best F1: 0.8374 at epoch 20. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 12:03:03,734 [INFO] Fold 3 Phase 2 - Epoch 14/20 (Total: 21), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1916, Val Loss: 0.2702, Val Acc: 0.9188, Val F1: 0.8386, Val AUC: 0.9508, Time: 273.42s
2025-05-11 12:03:08,345 [INFO]   -> New best F1: 0.8386 at epoch 21. Checkpoint saved to models_checkpointed/best_model_fold_3.pth
2025-05-11 12:07:42,228 [INFO] Fold 3 Phase 2 - Epoch 15/20 (Total: 22), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1904, Val Loss: 0.2672, Val Acc: 0.9188, Val F1: 0.8375, Val AUC: 0.9509, Time: 273.86s
2025-05-11 12:07:42,248 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 12:12:17,114 [INFO] Fold 3 Phase 2 - Epoch 16/20 (Total: 23), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1889, Val Loss: 0.2670, Val Acc: 0.9188, Val F1: 0.8372, Val AUC: 0.9508, Time: 274.85s
2025-05-11 12:12:17,133 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 12:16:50,779 [INFO] Fold 3 Phase 2 - Epoch 17/20 (Total: 24), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1864, Val Loss: 0.2670, Val Acc: 0.9192, Val F1: 0.8382, Val AUC: 0.9512, Time: 273.63s
2025-05-11 12:16:50,799 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 12:21:26,055 [INFO] Fold 3 Phase 2 - Epoch 18/20 (Total: 25), LR: [1.0000000000000002e-07, 1e-05], Train Loss: 0.1861, Val Loss: 0.2667, Val Acc: 0.9195, Val F1: 0.8386, Val AUC: 0.9514, Time: 275.24s
2025-05-11 12:21:26,076 [INFO]   -> F1 did not improve. Patience: 4/7
2025-05-11 12:26:01,669 [INFO] Fold 3 Phase 2 - Epoch 19/20 (Total: 26), LR: [1.0000000000000004e-08, 1.0000000000000002e-06], Train Loss: 0.1857, Val Loss: 0.2668, Val Acc: 0.9192, Val F1: 0.8380, Val AUC: 0.9514, Time: 275.57s
2025-05-11 12:26:01,689 [INFO]   -> F1 did not improve. Patience: 5/7
2025-05-11 12:30:37,495 [INFO] Fold 3 Phase 2 - Epoch 20/20 (Total: 27), LR: [1.0000000000000004e-08, 1.0000000000000002e-06], Train Loss: 0.1804, Val Loss: 0.2671, Val Acc: 0.9192, Val F1: 0.8380, Val AUC: 0.9514, Time: 275.78s
2025-05-11 12:30:37,516 [INFO]   -> F1 did not improve. Patience: 6/7
2025-05-11 12:30:37,537 [INFO] Loading best model from models_checkpointed/best_model_fold_3.pth (Epoch 21, F1: 0.8386)
2025-05-11 12:30:38,472 [INFO] --- Evaluating Best Model for Fold 3 ---
2025-05-11 12:30:55,566 [INFO] Fold 3 Final Validation Results (Best Model):
2025-05-11 12:30:55,587 [INFO]   Accuracy:  0.9188
2025-05-11 12:30:55,608 [INFO]   F1 Score:  0.8386
2025-05-11 12:30:55,629 [INFO]   Precision: 0.8381
2025-05-11 12:30:55,650 [INFO]   Recall:    0.8392
2025-05-11 12:30:55,672 [INFO]   AUC:       0.9508
2025-05-11 12:30:55,693 [INFO]   Loss:      0.2702
2025-05-11 12:30:55,714 [INFO]   Confusion Matrix:
2025-05-11 12:30:55,735 [INFO] 
[[4065  234]
 [ 232 1211]]
2025-05-11 12:30:55,756 [INFO] --- Fold 3 completed in 6328.87s ---
2025-05-11 12:30:55,777 [INFO] --- Starting Fold 4/5 ---
2025-05-11 12:30:55,800 [INFO] Initializing VGG11 model.
2025-05-11 12:30:56,877 [INFO] Adjusted VGG11's existing dropout rates at classifier indices [2, 5] to 0.5
2025-05-11 12:30:56,883 [INFO] Replaced final layer of VGG11 (classifier[6]) for 2 output classes.
2025-05-11 12:30:57,047 [INFO] --- Phase 1: Training Head ---
2025-05-11 12:33:50,303 [INFO] Fold 4 Phase 1 - Epoch 1/7, Train Loss: 0.6789, Time: 173.23s
2025-05-11 12:36:44,448 [INFO] Fold 4 Phase 1 - Epoch 2/7, Train Loss: 0.5829, Time: 174.12s
2025-05-11 12:39:39,662 [INFO] Fold 4 Phase 1 - Epoch 3/7, Train Loss: 0.5283, Time: 175.19s
2025-05-11 12:42:34,529 [INFO] Fold 4 Phase 1 - Epoch 4/7, Train Loss: 0.5101, Time: 174.85s
2025-05-11 12:45:30,045 [INFO] Fold 4 Phase 1 - Epoch 5/7, Train Loss: 0.5801, Time: 175.49s
2025-05-11 12:48:25,903 [INFO] Fold 4 Phase 1 - Epoch 6/7, Train Loss: 0.6202, Time: 175.84s
2025-05-11 12:51:22,861 [INFO] Fold 4 Phase 1 - Epoch 7/7, Train Loss: 0.5093, Time: 176.94s
2025-05-11 12:51:22,885 [INFO] --- Phase 2: Fine-tuning Full Model ---
2025-05-11 12:55:58,849 [INFO] Fold 4 Phase 2 - Epoch 1/20 (Total: 8), LR: [1e-05, 0.001], Train Loss: 0.5385, Val Loss: 0.3849, Val Acc: 0.8680, Val F1: 0.7100, Val AUC: 0.8897, Time: 275.94s
2025-05-11 12:56:00,524 [INFO]   -> New best F1: 0.7100 at epoch 8. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:00:36,526 [INFO] Fold 4 Phase 2 - Epoch 2/20 (Total: 9), LR: [1e-05, 0.001], Train Loss: 0.4336, Val Loss: 0.3280, Val Acc: 0.8878, Val F1: 0.7407, Val AUC: 0.9140, Time: 275.98s
2025-05-11 13:00:41,103 [INFO]   -> New best F1: 0.7407 at epoch 9. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:05:17,028 [INFO] Fold 4 Phase 2 - Epoch 3/20 (Total: 10), LR: [1e-05, 0.001], Train Loss: 0.3665, Val Loss: 0.3332, Val Acc: 0.8865, Val F1: 0.7899, Val AUC: 0.9325, Time: 275.90s
2025-05-11 13:05:21,711 [INFO]   -> New best F1: 0.7899 at epoch 10. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:10:00,188 [INFO] Fold 4 Phase 2 - Epoch 4/20 (Total: 11), LR: [1e-05, 0.001], Train Loss: 0.3334, Val Loss: 0.3297, Val Acc: 0.8894, Val F1: 0.7939, Val AUC: 0.9385, Time: 278.45s
2025-05-11 13:10:04,916 [INFO]   -> New best F1: 0.7939 at epoch 11. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:14:42,596 [INFO] Fold 4 Phase 2 - Epoch 5/20 (Total: 12), LR: [1e-05, 0.001], Train Loss: 0.3135, Val Loss: 0.3088, Val Acc: 0.8985, Val F1: 0.8010, Val AUC: 0.9404, Time: 277.66s
2025-05-11 13:14:47,249 [INFO]   -> New best F1: 0.8010 at epoch 12. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:19:26,518 [INFO] Fold 4 Phase 2 - Epoch 6/20 (Total: 13), LR: [1e-05, 0.001], Train Loss: 0.2976, Val Loss: 0.2860, Val Acc: 0.8985, Val F1: 0.8011, Val AUC: 0.9377, Time: 279.24s
2025-05-11 13:19:31,248 [INFO]   -> New best F1: 0.8011 at epoch 13. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:24:18,603 [INFO] Fold 4 Phase 2 - Epoch 7/20 (Total: 14), LR: [1e-05, 0.001], Train Loss: 0.2938, Val Loss: 0.2755, Val Acc: 0.8999, Val F1: 0.8099, Val AUC: 0.9481, Time: 287.33s
2025-05-11 13:24:23,248 [INFO]   -> New best F1: 0.8099 at epoch 14. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:29:07,447 [INFO] Fold 4 Phase 2 - Epoch 8/20 (Total: 15), LR: [1e-05, 0.001], Train Loss: 0.2749, Val Loss: 0.2575, Val Acc: 0.9058, Val F1: 0.8200, Val AUC: 0.9525, Time: 284.17s
2025-05-11 13:29:12,052 [INFO]   -> New best F1: 0.8200 at epoch 15. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:33:59,374 [INFO] Fold 4 Phase 2 - Epoch 9/20 (Total: 16), LR: [1e-05, 0.001], Train Loss: 0.2658, Val Loss: 0.2556, Val Acc: 0.9152, Val F1: 0.8243, Val AUC: 0.9472, Time: 287.30s
2025-05-11 13:34:04,022 [INFO]   -> New best F1: 0.8243 at epoch 16. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:38:52,571 [INFO] Fold 4 Phase 2 - Epoch 10/20 (Total: 17), LR: [1e-05, 0.001], Train Loss: 0.2505, Val Loss: 0.2550, Val Acc: 0.9114, Val F1: 0.8259, Val AUC: 0.9501, Time: 288.52s
2025-05-11 13:38:57,241 [INFO]   -> New best F1: 0.8259 at epoch 17. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 13:43:46,560 [INFO] Fold 4 Phase 2 - Epoch 11/20 (Total: 18), LR: [1e-05, 0.001], Train Loss: 0.2399, Val Loss: 0.2662, Val Acc: 0.9124, Val F1: 0.8258, Val AUC: 0.9499, Time: 289.29s
2025-05-11 13:43:46,585 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 13:48:37,074 [INFO] Fold 4 Phase 2 - Epoch 12/20 (Total: 19), LR: [1e-05, 0.001], Train Loss: 0.2294, Val Loss: 0.2816, Val Acc: 0.9075, Val F1: 0.8157, Val AUC: 0.9446, Time: 290.46s
2025-05-11 13:48:37,100 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 13:53:28,982 [INFO] Fold 4 Phase 2 - Epoch 13/20 (Total: 20), LR: [1e-05, 0.001], Train Loss: 0.2252, Val Loss: 0.3385, Val Acc: 0.9067, Val F1: 0.8013, Val AUC: 0.9471, Time: 291.86s
2025-05-11 13:53:29,008 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 13:58:21,251 [INFO] Fold 4 Phase 2 - Epoch 14/20 (Total: 21), LR: [1e-05, 0.001], Train Loss: 0.2172, Val Loss: 0.3040, Val Acc: 0.9180, Val F1: 0.8300, Val AUC: 0.9505, Time: 292.22s
2025-05-11 13:58:25,918 [INFO]   -> New best F1: 0.8300 at epoch 21. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 14:03:20,004 [INFO] Fold 4 Phase 2 - Epoch 15/20 (Total: 22), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1814, Val Loss: 0.2538, Val Acc: 0.9178, Val F1: 0.8355, Val AUC: 0.9553, Time: 294.06s
2025-05-11 14:03:24,739 [INFO]   -> New best F1: 0.8355 at epoch 22. Checkpoint saved to models_checkpointed/best_model_fold_4.pth
2025-05-11 14:08:18,296 [INFO] Fold 4 Phase 2 - Epoch 16/20 (Total: 23), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1725, Val Loss: 0.2568, Val Acc: 0.9169, Val F1: 0.8336, Val AUC: 0.9560, Time: 293.53s
2025-05-11 14:08:18,324 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 14:13:13,187 [INFO] Fold 4 Phase 2 - Epoch 17/20 (Total: 24), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1662, Val Loss: 0.2629, Val Acc: 0.9169, Val F1: 0.8319, Val AUC: 0.9555, Time: 294.84s
2025-05-11 14:13:13,215 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 14:18:12,256 [INFO] Fold 4 Phase 2 - Epoch 18/20 (Total: 25), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1627, Val Loss: 0.2538, Val Acc: 0.9166, Val F1: 0.8318, Val AUC: 0.9565, Time: 299.01s
2025-05-11 14:18:12,283 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 14:23:12,744 [INFO] Fold 4 Phase 2 - Epoch 19/20 (Total: 26), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1631, Val Loss: 0.2619, Val Acc: 0.9166, Val F1: 0.8320, Val AUC: 0.9557, Time: 300.43s
2025-05-11 14:23:12,774 [INFO]   -> F1 did not improve. Patience: 4/7
2025-05-11 14:28:11,954 [INFO] Fold 4 Phase 2 - Epoch 20/20 (Total: 27), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1550, Val Loss: 0.2695, Val Acc: 0.9166, Val F1: 0.8297, Val AUC: 0.9538, Time: 299.15s
2025-05-11 14:28:11,983 [INFO]   -> F1 did not improve. Patience: 5/7
2025-05-11 14:28:12,011 [INFO] Loading best model from models_checkpointed/best_model_fold_4.pth (Epoch 22, F1: 0.8355)
2025-05-11 14:28:12,983 [INFO] --- Evaluating Best Model for Fold 4 ---
2025-05-11 14:28:32,260 [INFO] Fold 4 Final Validation Results (Best Model):
2025-05-11 14:28:32,288 [INFO]   Accuracy:  0.9178
2025-05-11 14:28:32,316 [INFO]   F1 Score:  0.8355
2025-05-11 14:28:32,345 [INFO]   Precision: 0.8402
2025-05-11 14:28:32,374 [INFO]   Recall:    0.8309
2025-05-11 14:28:32,403 [INFO]   AUC:       0.9553
2025-05-11 14:28:32,431 [INFO]   Loss:      0.2538
2025-05-11 14:28:32,459 [INFO]   Confusion Matrix:
2025-05-11 14:28:32,487 [INFO] 
[[4071  228]
 [ 244 1199]]
2025-05-11 14:28:32,516 [INFO] --- Fold 4 completed in 7056.72s ---
2025-05-11 14:28:32,544 [INFO] --- Starting Fold 5/5 ---
2025-05-11 14:28:32,576 [INFO] Initializing VGG11 model.
2025-05-11 14:28:33,729 [INFO] Adjusted VGG11's existing dropout rates at classifier indices [2, 5] to 0.5
2025-05-11 14:28:33,738 [INFO] Replaced final layer of VGG11 (classifier[6]) for 2 output classes.
2025-05-11 14:28:33,897 [INFO] --- Phase 1: Training Head ---
2025-05-11 14:31:51,110 [INFO] Fold 5 Phase 1 - Epoch 1/7, Train Loss: 0.7093, Time: 197.18s
2025-05-11 14:35:10,783 [INFO] Fold 5 Phase 1 - Epoch 2/7, Train Loss: 0.5434, Time: 199.64s
2025-05-11 14:38:31,862 [INFO] Fold 5 Phase 1 - Epoch 3/7, Train Loss: 0.5213, Time: 201.05s
2025-05-11 14:41:53,103 [INFO] Fold 5 Phase 1 - Epoch 4/7, Train Loss: 0.5084, Time: 201.21s
2025-05-11 14:45:14,411 [INFO] Fold 5 Phase 1 - Epoch 5/7, Train Loss: 0.5176, Time: 201.28s
2025-05-11 14:48:35,729 [INFO] Fold 5 Phase 1 - Epoch 6/7, Train Loss: 0.5120, Time: 201.29s
2025-05-11 14:51:57,109 [INFO] Fold 5 Phase 1 - Epoch 7/7, Train Loss: 0.5663, Time: 201.35s
2025-05-11 14:51:57,138 [INFO] --- Phase 2: Fine-tuning Full Model ---
2025-05-11 14:57:02,402 [INFO] Fold 5 Phase 2 - Epoch 1/20 (Total: 8), LR: [1e-05, 0.001], Train Loss: 0.5742, Val Loss: 0.4684, Val Acc: 0.8420, Val F1: 0.6667, Val AUC: 0.8677, Time: 305.23s
2025-05-11 14:57:04,150 [INFO]   -> New best F1: 0.6667 at epoch 8. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:02:08,478 [INFO] Fold 5 Phase 2 - Epoch 2/20 (Total: 9), LR: [1e-05, 0.001], Train Loss: 0.4747, Val Loss: 0.3651, Val Acc: 0.8645, Val F1: 0.7426, Val AUC: 0.9059, Time: 304.30s
2025-05-11 15:02:12,887 [INFO]   -> New best F1: 0.7426 at epoch 9. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:07:19,369 [INFO] Fold 5 Phase 2 - Epoch 3/20 (Total: 10), LR: [1e-05, 0.001], Train Loss: 0.3628, Val Loss: 0.3124, Val Acc: 0.8845, Val F1: 0.7740, Val AUC: 0.9249, Time: 306.45s
2025-05-11 15:07:23,823 [INFO]   -> New best F1: 0.7740 at epoch 10. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:12:29,316 [INFO] Fold 5 Phase 2 - Epoch 4/20 (Total: 11), LR: [1e-05, 0.001], Train Loss: 0.3406, Val Loss: 0.2910, Val Acc: 0.8913, Val F1: 0.7736, Val AUC: 0.9277, Time: 305.46s
2025-05-11 15:12:29,347 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 15:17:36,007 [INFO] Fold 5 Phase 2 - Epoch 5/20 (Total: 12), LR: [1e-05, 0.001], Train Loss: 0.3083, Val Loss: 0.2883, Val Acc: 0.8932, Val F1: 0.7838, Val AUC: 0.9317, Time: 306.63s
2025-05-11 15:17:40,566 [INFO]   -> New best F1: 0.7838 at epoch 12. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:22:47,682 [INFO] Fold 5 Phase 2 - Epoch 6/20 (Total: 13), LR: [1e-05, 0.001], Train Loss: 0.2987, Val Loss: 0.2811, Val Acc: 0.8984, Val F1: 0.7732, Val AUC: 0.9394, Time: 307.08s
2025-05-11 15:22:47,713 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 15:27:55,237 [INFO] Fold 5 Phase 2 - Epoch 7/20 (Total: 14), LR: [1e-05, 0.001], Train Loss: 0.2771, Val Loss: 0.2822, Val Acc: 0.8972, Val F1: 0.8001, Val AUC: 0.9408, Time: 307.49s
2025-05-11 15:28:00,290 [INFO]   -> New best F1: 0.8001 at epoch 14. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:33:07,531 [INFO] Fold 5 Phase 2 - Epoch 8/20 (Total: 15), LR: [1e-05, 0.001], Train Loss: 0.2606, Val Loss: 0.2757, Val Acc: 0.8983, Val F1: 0.8044, Val AUC: 0.9464, Time: 307.21s
2025-05-11 15:33:11,893 [INFO]   -> New best F1: 0.8044 at epoch 15. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:38:19,718 [INFO] Fold 5 Phase 2 - Epoch 9/20 (Total: 16), LR: [1e-05, 0.001], Train Loss: 0.2592, Val Loss: 0.2644, Val Acc: 0.9012, Val F1: 0.8112, Val AUC: 0.9491, Time: 307.79s
2025-05-11 15:38:24,285 [INFO]   -> New best F1: 0.8112 at epoch 16. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 15:43:32,098 [INFO] Fold 5 Phase 2 - Epoch 10/20 (Total: 17), LR: [1e-05, 0.001], Train Loss: 0.2409, Val Loss: 0.2562, Val Acc: 0.9075, Val F1: 0.8102, Val AUC: 0.9446, Time: 307.78s
2025-05-11 15:43:32,131 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 15:48:40,130 [INFO] Fold 5 Phase 2 - Epoch 11/20 (Total: 18), LR: [1e-05, 0.001], Train Loss: 0.2298, Val Loss: 0.2794, Val Acc: 0.8962, Val F1: 0.8048, Val AUC: 0.9470, Time: 307.97s
2025-05-11 15:48:40,163 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 15:53:48,310 [INFO] Fold 5 Phase 2 - Epoch 12/20 (Total: 19), LR: [1e-05, 0.001], Train Loss: 0.2273, Val Loss: 0.2552, Val Acc: 0.9007, Val F1: 0.8099, Val AUC: 0.9490, Time: 308.11s
2025-05-11 15:53:48,343 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 15:58:56,920 [INFO] Fold 5 Phase 2 - Epoch 13/20 (Total: 20), LR: [1e-05, 0.001], Train Loss: 0.2116, Val Loss: 0.2741, Val Acc: 0.9075, Val F1: 0.8124, Val AUC: 0.9450, Time: 308.54s
2025-05-11 15:59:01,444 [INFO]   -> New best F1: 0.8124 at epoch 20. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 16:04:10,787 [INFO] Fold 5 Phase 2 - Epoch 14/20 (Total: 21), LR: [1e-05, 0.001], Train Loss: 0.2050, Val Loss: 0.2589, Val Acc: 0.9140, Val F1: 0.8280, Val AUC: 0.9510, Time: 309.31s
2025-05-11 16:04:15,311 [INFO]   -> New best F1: 0.8280 at epoch 21. Checkpoint saved to models_checkpointed/best_model_fold_5.pth
2025-05-11 16:09:24,118 [INFO] Fold 5 Phase 2 - Epoch 15/20 (Total: 22), LR: [1e-05, 0.001], Train Loss: 0.1994, Val Loss: 0.2864, Val Acc: 0.9099, Val F1: 0.8214, Val AUC: 0.9491, Time: 308.77s
2025-05-11 16:09:24,151 [INFO]   -> F1 did not improve. Patience: 1/7
2025-05-11 16:14:34,156 [INFO] Fold 5 Phase 2 - Epoch 16/20 (Total: 23), LR: [1e-05, 0.001], Train Loss: 0.1901, Val Loss: 0.3004, Val Acc: 0.9049, Val F1: 0.8112, Val AUC: 0.9449, Time: 309.97s
2025-05-11 16:14:34,190 [INFO]   -> F1 did not improve. Patience: 2/7
2025-05-11 16:19:45,750 [INFO] Fold 5 Phase 2 - Epoch 17/20 (Total: 24), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1622, Val Loss: 0.2863, Val Acc: 0.9087, Val F1: 0.8175, Val AUC: 0.9501, Time: 311.53s
2025-05-11 16:19:45,785 [INFO]   -> F1 did not improve. Patience: 3/7
2025-05-11 16:24:59,075 [INFO] Fold 5 Phase 2 - Epoch 18/20 (Total: 25), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1425, Val Loss: 0.2927, Val Acc: 0.9087, Val F1: 0.8189, Val AUC: 0.9504, Time: 313.26s
2025-05-11 16:24:59,109 [INFO]   -> F1 did not improve. Patience: 4/7
2025-05-11 16:30:14,355 [INFO] Fold 5 Phase 2 - Epoch 19/20 (Total: 26), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1398, Val Loss: 0.3045, Val Acc: 0.9091, Val F1: 0.8226, Val AUC: 0.9512, Time: 315.21s
2025-05-11 16:30:14,390 [INFO]   -> F1 did not improve. Patience: 5/7
2025-05-11 16:35:30,087 [INFO] Fold 5 Phase 2 - Epoch 20/20 (Total: 27), LR: [1.0000000000000002e-06, 0.0001], Train Loss: 0.1369, Val Loss: 0.3092, Val Acc: 0.9096, Val F1: 0.8233, Val AUC: 0.9507, Time: 315.66s
2025-05-11 16:35:30,122 [INFO]   -> F1 did not improve. Patience: 6/7
2025-05-11 16:35:30,158 [INFO] Loading best model from models_checkpointed/best_model_fold_5.pth (Epoch 21, F1: 0.8280)
2025-05-11 16:35:31,167 [INFO] --- Evaluating Best Model for Fold 5 ---
2025-05-11 16:35:52,004 [INFO] Fold 5 Final Validation Results (Best Model):
2025-05-11 16:35:52,039 [INFO]   Accuracy:  0.9140
2025-05-11 16:35:52,075 [INFO]   F1 Score:  0.8280
2025-05-11 16:35:52,110 [INFO]   Precision: 0.8321
2025-05-11 16:35:52,145 [INFO]   Recall:    0.8240
2025-05-11 16:35:52,180 [INFO]   AUC:       0.9510
2025-05-11 16:35:52,216 [INFO]   Loss:      0.2589
2025-05-11 16:35:52,251 [INFO]   Confusion Matrix:
2025-05-11 16:35:52,287 [INFO] 
[[4058  240]
 [ 254 1189]]
2025-05-11 16:35:52,323 [INFO] --- Fold 5 completed in 7639.75s ---

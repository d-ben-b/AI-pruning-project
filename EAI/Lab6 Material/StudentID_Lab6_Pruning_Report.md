# Lab 6 - Transformer Pruning Report
  
1. **請說明 get_real_idx 實作部分是怎麼做的** 10%

The `get_real_idx` function tracks the original image patch indices through the trimming process. Since pruning reduces the number of tokens, the indices in deeper layers are relative to the *reduced* sequence from the previous layer. 
- It takes the indices of tokens kept in the current layer and uses `torch.gather` to select from the list of *original* indices kept by the previous layer (`prev_real_idx`). This chains the mapping back to the absolute position in the original image.
- It handles **fused tokens** (if enabled) by concatenating a dummy index (0, representing the top-left patch) to the source list before gathering. This ensures the fused token is preserved in the visualization at a fixed position.

2. **實際在哪些層做了 pruning ?** 10%
    
Pruning is performed at **Block 3, Block 6, and Block 9** (0-indexed). This corresponds to the layers configured with a `keep_rate` of 0.7 in the `EViT` initialization tuple: `(1, 1, 1, 0.7) + (1, 1, 0.7) + (1, 1, 0.7) + (1, 1)`.
    
3. **如果沒有 get_real_idx 可視化結果會長怎樣，為什麼 ?** 10%
    
Without `get_real_idx`, the visualization would be **incorrect and scrambled**. The visualizer would interpret the relative indices (e.g., 0 to N_kept) as if they were absolute spatial coordinates in the original image. This would result in the kept patches always clustering at the top-left of the image (indices 0 to N) regardless of their actual semantic position, failing to show which parts of the object were actually preserved.
    
4. **分析視覺化的圖，這些變化代表著什麼 ?** 10%
    
The visualization demonstrates that the attention-based pruning policy successfully preserves the **foreground object** (semantically important regions like the dog or plane) while discarding **redundant background patches** (sky, grass, flat textures). This indicates the model learns to attend to salient features and can efficiently reduce computation by removing non-informative regions.


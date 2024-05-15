# Experiments
1. train only control net
2. train only unet
3. train one after the other
4. train with "cheat" scenario (probably upper bound)
    i.e., as previous frame give the ground truth frame to the unet
5. test the above also with the v1.0 of the image variations model
7. code for quantitative results
8. if both v1 and v2 image variation models seem to not train, think how to use
    the base sd1.5 to finetune
9. for the most promising model, run small parameter sweep for guidance scale,
    ip-adapter (plus or not) and ip-adapter scale
10. LSTM or Video Transformers (ViViT/TimeSformer) -- https://huggingface.co/docs/transformers/model_doc/timesformer
11. RamVid etc. (?)

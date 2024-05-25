# Experiments 1.0
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

# Experiments 2.0
We probably fixed a bag in the dataset transforms and preperation

## Grid Search
- models = [`lambdalabs/sd-image-variations-diffusers`, `runwayml/stable-diffusion-v1-5`]
- cheat = [`True`, `False`]
- train_unet = [`True`, `False`]
- use_control_net = [`True`, `False`]
- train_control_net = [`True`, `False`]
- use_mapper = [`True`, `False`]

> I expect the mapper to be more important for the SD1.5 (especially if it is trained first)

> Remember to use `antialias=False` for the image encoder transforms if Image Variations model is used

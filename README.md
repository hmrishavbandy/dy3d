# Doodle Your 3D: From Abstract Freehand Sketches to Precise 3D Shapes [CVPR 2024]

![Banner GIF](/assets/banner.gif)

In this paper, we democratise 3D content creation, enabling precise generation of 3D shapes from abstract sketches while overcoming limitations tied to drawing skills. We introduce a novel part-level modelling and alignment framework that facilitates abstraction modelling and cross-modal correspondence. Leveraging the same part-level decoder, our approach seamlessly extends to sketch modelling by establishing correspondence between CLIPasso edgemaps and projected 3D part regions, eliminating the need for a dataset pairing human sketches and 3D shapes. Additionally, our method introduces a seamless in-position editing process as a byproduct of cross-modal part-aligned modelling. Operating in a low-dimensional implicit space, our approach significantly reduces computational demands and processing time.


### Dataset: 
We synthesize our own dataset by :
- Inverting each shape with [SPAGHETTI](https://github.com/amirhertz/spaghetti), which gives us a single latent $\in \mathbb{R}^{m\times d}$ for each shape. Each latent consists of $m$ features that correspond to $m$ parts that the shape has been divided into.
- We align these latents so that they have shape-parts similarly indexed - as `samples_final.npy`.
- Next, we create sketches of 3D shapes by rendering them with pre-defined cameras and processing them with [CLIPasso](https://clipasso.github.io/clipasso/).
- We segment 3D meshes with our part-disentangled latent representation, and render these segments with same camera positions. This gives us a segment map overlay over each sketch.

Sample data is available in `data/sample/`



### Training

The training is a two stage process: first the sketch segmentation network is trained and next, the diffusion model for conditional shape generation is trained. To train these networks, the dataset has to be created: in the form of sketches of 3D shape projections, and their segment maps.

To train the segmentation network, run:

```
sh scripts/segment_train.sh
```

To precompute sketch encodings (for fast diffusion model training), run:

```
sh scripts/precompute.sh
```

To train the diffusion model, run:

```
sh scripts/diffusion_train.sh
```

To scale training, change the `size` parameter of the diffusion model. As per my pre-liminary tests, increasing `size` to 1024 results in a 73M param diffusion model, but the shape quality goes up.
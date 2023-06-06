# Notes

## Always the same prediction

May be caused
- by overfitting of rec_loss to reconstruct the input 1:1
- by hidden node embedding and random generation of it

We can try
+ save hidden node embedding value in the checkpoint and load it
- check accuracy on hidden nodes only
- modify loss to **focus on reconstructing hidden nodes** from neighbors (need investivation)?
- use ConceptNet since it might have more informative neighbors for IdenProf task
+ test embeddings on VG (reconstruction)
- repeated <person> <is> <profession> triples (voting for profession prediction)
- (!) use CosSim in loss (remove classification)

Check accuracy on the hidden nodes only during training, different metric, losses
Chech the training setup in general

27-29th Sep

LaTeX
(min 60 pages in total)

Introduction (Problem statement, motivation, abstract of)
(!) Literature review (first start with that), SG pred/gen, papers on human role pred, SG, Self-/Un-Supervised learning
  Topic
    papers
    papers (story, topics connected together)
  Topic
    paper
Background
Methodology
Experiments
+
Results (15 pages)
Conclusion

1. Look at article on Message Passing on Graphs
2. Look at how can we improve MP in our case
3. Fix evaluation scripts

Explore attention coefficients

Train w/o Neighbor nodes, eval with 2nn
different propagation techniques
toy data to test
ConceptNet

https://www.google.ru/maps/place/printy+Digitaldruck+M%C3%BCnchen/@48.1481958,11.5608768,15.73z/data=!4m10!1m2!2m1!1sprinty!3m6!1s0x479e75e69c96a991:0xea81efdb9b05a61b!8m2!3d48.149688!4d11.566133!15sCgZwcmludHlaCCIGcHJpbnR5kgEKcHJpbnRfc2hvcOABAA!16s%2Fg%2F1tj83vg2

printy

1 to try: For a given scene graph: compare smallest-dist to each of the prof nodes.

Chair meeting with Azade and students - Friday, 5 May, 13:30

For the IdenProf:
for each image:
calculate the dist to all the classes

---
calculate avg distance (confusion matrix)
gt class \ classes

Generate Scene Graphs with https://github.com/JackWhite-rwx/SceneGraphGenZeroShotWithGSAM
and compare the confusion matrix


Results

Datasets used
Hyperparameters, NN architectures

1. Supervised with IdenProf
2. CLIP results

3. Scene Generation results with pretrained on VG, with zero-shot (SceneGraphGenZeroShotWithGSAM)

4. Iterative results (adding visual features, masking)

5. Heatmaps, conf matrix

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

# To do

1. Test with the provided model of reconstruction

compare evaluation: eval.py (fixed)
compare wn_n

2. Debug the issue with training DSGAE model (-)
3. Look at the previous model (acc on VG)
4. Try to extend graphs with WordNet in DSGAE repository (out of GPU memory)

Consider L2 or Hinge loss
Try training for 50000 iter

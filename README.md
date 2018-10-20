# Awesome Visual Grounding
A curated list of research papers in grounding. Link to the code if available is also present.

Visual grounding task refers to localizing an object given a query or a sentence. It is somewhat related to Visual Question Answering so this repository might also help [https://github.com/JamesChuanggg/awesome-vqa](https://github.com/JamesChuanggg/awesome-vqa)

## Contributing
Feel free to contact me [theshadow29.github.io](theshadow29.github.io) or open an issue or submit a pull request. 

## Datasets
### Image Grounding Datasets

1. Plummer, Bryan A., et al. **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** Proceedings of the IEEE international conference on computer vision. 2015. [[Paper]](https://arxiv.org/abs/1505.04870) [[Code]](https://github.com/BryanPlummer/pl-clc) [[Website]](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)

2. Kazemzadeh, Sahar, et al. **Referitgame: Referring to objects in photographs of natural scenes.** Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014. [[Paper]](http://www.aclweb.org/anthology/D14-1086) [[Website]](http://tamaraberg.com/referitgame/)

3. Krishna, Ranjay, et al. **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** International Journal of Computer Vision 123.1 (2017): 32-73. [[Paper]](https://arxiv.org/pdf/1602.07332.pdf) [[Website]](https://visualgenome.org/)

### Video Datasets
To be updated

## Paper Roadmap (Chronological Order):

### Visual Grounding (Images):
1. Karpathy, Andrej, Armand Joulin, and Li F. Fei-Fei. **Deep fragment embeddings for bidirectional image sentence mapping.** Advances in neural information processing systems. 2014. [[Paper]](http://papers.nips.cc/paper/5281-deep-fragment-embeddings-for-bidirectional-image-sentence-mapping.pdf)
2. Karpathy, Andrej, and Li Fei-Fei. **Deep visual-semantic alignments for generating image descriptions.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. *Method name: Neural Talk*. [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf) [[Code]](https://github.com/karpathy/neuraltalk) [[Torch Code]](https://github.com/karpathy/neuraltalk2) [[Website]](https://cs.stanford.edu/people/karpathy/deepimagesent/) 
3. Hu, Ronghang, et al. **Natural language object retrieval.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. *Method name: Spatial Context Recurrent
ConvNet (SCRC)* [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hu_Natural_Language_Object_CVPR_2016_paper.pdf) [[Code]](https://github.com/ronghanghu/natural-language-object-retrieval) [[Website]](http://ronghanghu.com/text_obj_retrieval/)
4. Rohrbach, Anna, et al. **Grounding of textual phrases in images by reconstruction.** European Conference on Computer Vision. Springer, Cham, 2016. *Method Name: GroundR* [[Paper]](https://arxiv.org/pdf/1511.03745.pdf) [[Tensorflow Code]](https://github.com/kanchen-usc/GroundeR) [[Torch Code]](https://github.com/ruotianluo/refexp-comprehension)
5. Wang, Mingzhe, et al. **Structured matching for phrase localization.** European Conference on Computer Vision. Springer, Cham, 2016. *Method name: Structured Matching* [[Paper]](https://pdfs.semanticscholar.org/9216/2ec88ad974cc5082d9688c8bfee672ad59ad.pdf) [[Code]](https://github.com/princeton-vl/structured-matching)
6. Fukui, Akira et al. **Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding.** EMNLP (2016). *Method name: MCB* [[Paper]](https://arxiv.org/pdf/1606.01847.pdf)[[Code]](https://github.com/akirafukui/vqa-mcb)
7. Wu, Fan et al. **An End-to-End Approach to Natural Language Object Retrieval via Context-Aware Deep Reinforcement Learning.** CoRR abs/1703.07579 (2017): n. pag. [[Paper]](https://arxiv.org/pdf/1703.07579.pdf) [[Code]](https://github.com/jxwufan/NLOR_A3C)
8. Chen, Kan, Rama Kovvuri, and Ram Nevatia. **Query-guided regression network with context policy for phrase grounding.** Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017. *Method name: QRC* [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Query-Guided_Regression_Network_ICCV_2017_paper.pdf) [[Code]](https://github.com/kanchen-usc/QRC-Net)
9. Li, Jianan, et al. **Deep attribute-preserving metric learning for natural language object retrieval.** Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017. [[Paper: ACM Link]](https://dl.acm.org/citation.cfm?id=3123439)

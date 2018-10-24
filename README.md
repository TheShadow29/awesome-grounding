# Awesome Visual Grounding
A curated list of research papers in grounding. Link to the code if available is also present.

Visual grounding task refers to localizing an object given a query or a sentence. It is also sometimes called referring expression comprehension. Referring expression is basically uniquely identifying the object in question. I have not included papers which do only referring expression generation, however if they also do the comprehension (or only comprehension) they have been included.

This task is somewhat related to Visual Question Answering so this repository might also help [https://github.com/JamesChuanggg/awesome-vqa](https://github.com/JamesChuanggg/awesome-vqa).

To maintaing the quality of the repo, I have gone through all the listed papers at least once before adding them to ensure their relevance to grounding. However, I might have missed some paper(s) or added some irrelevant paper(s). Feel free to open an issue in that case. I will go through the paper and then add / remove it. 

Video Grounding is work in progress. Should be completed sometime end of November. 

## Contributing
 Feel free to contact me [theshadow29.github.io](theshadow29.github.io) or open an issue or submit a pull request. 

## Datasets
### Image Grounding Datasets

1. **Flickr30k**: Plummer, Bryan A., et al. **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** Proceedings of the IEEE international conference on computer vision. 2015. [[Paper]](https://arxiv.org/abs/1505.04870) [[Code]](https://github.com/BryanPlummer/pl-clc) [[Website]](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)

1. **RefClef**: Kazemzadeh, Sahar, et al. **Referitgame: Referring to objects in photographs of natural scenes.** Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014. [[Paper]](http://www.aclweb.org/anthology/D14-1086) [[Website]](http://tamaraberg.com/referitgame/)

1. **RefCOCOg**: Mao, Junhua, et al. **Generation and comprehension of unambiguous object descriptions.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [[Paper]](https://arxiv.org/pdf/1511.02283.pdf) [[Code]](https://github.com/mjhucla/Google_Refexp_toolbox)

1. **RefCOCO and RefCOCO+**: Yu, Licheng, et al. **Modeling context in referring expressions.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1608.00272.pdf)[[Code]](https://github.com/lichengunc/refer)

1. **Visual Genome**: Krishna, Ranjay, et al. **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** International Journal of Computer Vision 123.1 (2017): 32-73. [[Paper]](https://arxiv.org/pdf/1602.07332.pdf) [[Website]](https://visualgenome.org/)

Instructions on RefClef, RefCOCO, RefCOCO+, RefCOCOg is nicely summarized here: https://github.com/lichengunc/refer

### Video Datasets
To be updated


## Paper Roadmap (Chronological Order):

### Visual Grounding / Referring Expressions (Images):
1. Karpathy, Andrej, Armand Joulin, and Li F. Fei-Fei. **Deep fragment embeddings for bidirectional image sentence mapping.** Advances in neural information processing systems. 2014. [[Paper]](http://papers.nips.cc/paper/5281-deep-fragment-embeddings-for-bidirectional-image-sentence-mapping.pdf)

1. Karpathy, Andrej, and Li Fei-Fei. **Deep visual-semantic alignments for generating image descriptions.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. *Method name: Neural Talk*. [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf) [[Code]](https://github.com/karpathy/neuraltalk) [[Torch Code]](https://github.com/karpathy/neuraltalk2) [[Website]](https://cs.stanford.edu/people/karpathy/deepimagesent/)

1. Hu, Ronghang, et al. **Natural language object retrieval.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. *Method name: Spatial Context Recurrent
ConvNet (SCRC)* [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hu_Natural_Language_Object_CVPR_2016_paper.pdf) [[Code]](https://github.com/ronghanghu/natural-language-object-retrieval) [[Website]](http://ronghanghu.com/text_obj_retrieval/)

1. Mao, Junhua, et al. **Generation and comprehension of unambiguous object descriptions.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [[Paper]](https://arxiv.org/pdf/1511.02283.pdf) [[Code]](https://github.com/mjhucla/Google_Refexp_toolbox)

1. Wang, Liwei, Yin Li, and Svetlana Lazebnik. **Learning deep structure-preserving image-text embeddings.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [[Paper]](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf) [[Code]](https://github.com/lwwang/Two_branch_network)

1. Yu, Licheng, et al. **Modeling context in referring expressions.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1608.00272.pdf)[[Code]](https://github.com/lichengunc/refer)

1. Nagaraja, Varun K., Vlad I. Morariu, and Larry S. Davis. **Modeling context between objects for referring expression understanding.** European Conference on Computer Vision. Springer, Cham, 2016.[[Paper]](https://arxiv.org/pdf/1608.00525.pdf) [[Code]](https://github.com/varun-nagaraja/referring-expressions)

1. Rohrbach, Anna, et al. **Grounding of textual phrases in images by reconstruction.** European Conference on Computer Vision. Springer, Cham, 2016. *Method Name: GroundR* [[Paper]](https://arxiv.org/pdf/1511.03745.pdf) [[Tensorflow Code]](https://github.com/kanchen-usc/GroundeR) [[Torch Code]](https://github.com/ruotianluo/refexp-comprehension)

1. Wang, Mingzhe, et al. **Structured matching for phrase localization.** European Conference on Computer Vision. Springer, Cham, 2016. *Method name: Structured Matching* [[Paper]](https://pdfs.semanticscholar.org/9216/2ec88ad974cc5082d9688c8bfee672ad59ad.pdf) [[Code]](https://github.com/princeton-vl/structured-matching)

1. Hu, Ronghang, Marcus Rohrbach, and Trevor Darrell. **Segmentation from natural language expressions.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1603.06180.pdf) [[Code]](https://github.com/ronghanghu/text_objseg) [[Website]](http://ronghanghu.com/text_objseg/)

1. Fukui, Akira et al. **Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding.** EMNLP (2016). *Method name: MCB* [[Paper]](https://arxiv.org/pdf/1606.01847.pdf)[[Code]](https://github.com/akirafukui/vqa-mcb)

1. Endo, Ko, et al. **An attention-based regression model for grounding textual phrases in images.** Proc. IJCAI. 2017. [[Paper]](https://www.ijcai.org/proceedings/2017/0558.pdf)

1. Chen, Kan, et al. **MSRC: Multimodal spatial regression with semantic context for phrase grounding.** International Journal of Multimedia Information Retrieval 7.1 (2018): 17-28. [[Paper -Springer Link]](https://link.springer.com/article/10.1007/s13735-017-0139-6)

1. Wu, Fan et al. **An End-to-End Approach to Natural Language Object Retrieval via Context-Aware Deep Reinforcement Learning.** CoRR abs/1703.07579 (2017): n. pag. [[Paper]](https://arxiv.org/pdf/1703.07579.pdf) [[Code]](https://github.com/jxwufan/NLOR_A3C)

1. Yu, Licheng, et al. **A joint speakerlistener-reinforcer model for referring expressions.** Computer Vision and Pattern Recognition (CVPR). Vol. 2. 2017. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_A_Joint_Speaker-Listener-Reinforcer_CVPR_2017_paper.pdf) [[Code]](https://github.com/lichengunc/speaker_listener_reinforcer)[[Website]](https://vision.cs.unc.edu/refer/)

1. Hu, Ronghang, et al. **Modeling relationships in referential expressions with compositional modular networks.** Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_Modeling_Relationships_in_CVPR_2017_paper.pdf) [[Code]](https://github.com/ronghanghu/cmn)

1. Luo, Ruotian, and Gregory Shakhnarovich. **Comprehension-guided referring expressions.** Computer Vision and Pattern Recognition (CVPR). Vol. 2. 2017. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Comprehension-Guided_Referring_Expressions_CVPR_2017_paper.pdf) [[Code]](https://github.com/ruotianluo/refexp-comprehension)

1. Liu, Jingyu, Liang Wang, and Ming-Hsuan Yang. **Referring expression generation and comprehension via attributes.** Proceedings of CVPR. 2017. [[Paper]](http://faculty.ucmerced.edu/mhyang/papers/iccv2017_referring_expression.pdf) 

1. Xiao, Fanyi, Leonid Sigal, and Yong Jae Lee. **Weakly-supervised visual grounding of phrases with linguistic structures.** arXiv preprint arXiv:1705.01371 (2017). [[Paper]](https://arxiv.org/pdf/1705.01371.pdf) 

1. Plummer, Bryan A., et al. **Phrase localization and visual relationship detection with comprehensive image-language cues.** Proc. ICCV. 2017. [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Plummer_Phrase_Localization_and_ICCV_2017_paper.pdf) [[Code]](https://github.com/BryanPlummer/pl-clc)

1. Chen, Kan, Rama Kovvuri, and Ram Nevatia. **Query-guided regression network with context policy for phrase grounding.** Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017. *Method name: QRC* [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Query-Guided_Regression_Network_ICCV_2017_paper.pdf) [[Code]](https://github.com/kanchen-usc/QRC-Net)

1. Liu, Chenxi, et al. **Recurrent Multimodal Interaction for Referring Image Segmentation.** ICCV. 2017. [[Paper]](https://arxiv.org/pdf/1703.07939.pdf) [[Code]](https://github.com/chenxi116/TF-phrasecut-public)

1. Li, Jianan, et al. **Deep attribute-preserving metric learning for natural language object retrieval.** Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017. [[Paper: ACM Link]](https://dl.acm.org/citation.cfm?id=3123439)

1. Li, Xiangyang, and Shuqiang Jiang. **Bundled Object Context for Referring Expressions.** IEEE Transactions on Multimedia (2018). [[Paper ieee link]](https://ieeexplore.ieee.org/document/8307406) 

1. Yu, Zhou, et al. **Rethinking Diversified and Discriminative Proposal Generation for Visual Grounding.** arXiv preprint arXiv:1805.03508 (2018). [[Paper]](https://www.ijcai.org/proceedings/2018/0155.pdf) [[Code]](https://github.com/XiangChenchao/DDPN)

1. Yu, Licheng, et al. **Mattnet: Modular attention network for referring expression comprehension.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_MAttNet_Modular_Attention_CVPR_2018_paper.pdf) [[Code]](https://github.com/lichengunc/MAttNet) [[Website]](http://vision2.cs.unc.edu/refer/comprehension)

1. Deng, Chaorui, et al. **Visual Grounding via Accumulated Attention.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Visual_Grounding_via_CVPR_2018_paper.pdf)

1. Zhang, Yundong, Juan Carlos Niebles, and Alvaro Soto. **Interpretable Visual Question Answering by Visual Grounding from Attention Supervision Mining.** arXiv preprint arXiv:1808.00265 (2018). [[Paper]](https://arxiv.org/pdf/1808.00265.pdf) 

1. Chen, Kan, Jiyang Gao, and Ram Nevatia. **Knowledge aided consistency for weakly supervised phrase grounding.** arXiv preprint arXiv:1803.03879 (2018). [[Paper]](https://arxiv.org/abs/1803.03879) [[Code]](https://github.com/kanchen-usc/KAC-Net)

1. Zhang, Hanwang, Yulei Niu, and Shih-Fu Chang. **Grounding referring expressions in images by variational context.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Grounding_Referring_Expressions_CVPR_2018_paper.pdf) [[Code]](https://github.com/yuleiniu/vc/)

1. Cirik, Volkan, Taylor Berg-Kirkpatrick, and Louis-Philippe Morency. **Using syntax to ground referring expressions in natural images.** arXiv preprint arXiv:1805.10547 (2018).[[Paper]](https://arxiv.org/pdf/1805.10547.pdf) [[Code]](https://github.com/volkancirik/groundnet)

### Object Retrieval

1. Hu, Ronghang, et al. **Natural language object retrieval.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. *Method name: Spatial Context Recurrent
ConvNet (SCRC)* [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hu_Natural_Language_Object_CVPR_2016_paper.pdf) [[Code]](https://github.com/ronghanghu/natural-language-object-retrieval) [[Website]](http://ronghanghu.com/text_obj_retrieval/)

1. Wu, Fan et al. **An End-to-End Approach to Natural Language Object Retrieval via Context-Aware Deep Reinforcement Learning.** CoRR abs/1703.07579 (2017): n. pag. [[Paper]](https://arxiv.org/pdf/1703.07579.pdf) [[Code]](https://github.com/jxwufan/NLOR_A3C)

1. Li, Jianan, et al. **Deep attribute-preserving metric learning for natural language object retrieval.** Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017. [[Paper: ACM Link]](https://dl.acm.org/citation.cfm?id=3123439)



# Awesome Visual Grounding
A curated list of research papers in grounding. Link to the code if available is also present.

Visual grounding task refers to localizing an object given a query or a sentence. It is also sometimes called referring expression comprehension. Referring expression is basically uniquely identifying the object in question. I have not included papers which do only referring expression generation, however if they also do the comprehension (or only comprehension) they have been included.

This task is somewhat related to Visual Question Answering so this repository might also help [https://github.com/JamesChuanggg/awesome-vqa](https://github.com/JamesChuanggg/awesome-vqa).

To maintaing the quality of the repo, I have gone through all the listed papers at least once before adding them to ensure their relevance to grounding. However, I might have missed some paper(s) or added some irrelevant paper(s). Feel free to open an issue in that case. I will go through the paper and then add / remove it. 

## Contributing
 Feel free to contact me [theshadow29.github.io](theshadow29.github.io) or open an issue or submit a pull request. 

## Datasets
### Image Grounding Datasets

1. **Flickr30k**: Plummer, Bryan A., et al. **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** Proceedings of the IEEE international conference on computer vision. 2015. [[Paper]](https://arxiv.org/abs/1505.04870) [[Code]](https://github.com/BryanPlummer/pl-clc) [[Website]](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)

1. **RefClef**: Kazemzadeh, Sahar, et al. **Referitgame: Referring to objects in photographs of natural scenes.** Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014. [[Paper]](http://www.aclweb.org/anthology/D14-1086) [[Website]](http://tamaraberg.com/referitgame/)

1. **RefCOCOg**: Mao, Junhua, et al. **Generation and comprehension of unambiguous object descriptions.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [[Paper]](https://arxiv.org/pdf/1511.02283.pdf) [[Code]](https://github.com/mjhucla/Google_Refexp_toolbox)

1. **RefCOCO and RefCOCO+**: 1. Yu, Licheng, et al. **Modeling context in referring expressions.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1608.00272.pdf)[[Code]](https://github.com/lichengunc/refer)

1. **Visual Genome**: Krishna, Ranjay, et al. **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** International Journal of Computer Vision 123.1 (2017): 32-73. [[Paper]](https://arxiv.org/pdf/1602.07332.pdf) [[Website]](https://visualgenome.org/)

Instructions on RefClef, RefCOCO, RefCOCO+, RefCOCOg is nicely summarized here: https://github.com/lichengunc/refer

### Video Datasets

1. **TaCoS**: Regneri, Michaela, et al. **Grounding action descriptions in videos.** Transactions of the Association of Computational Linguistics 1 (2013): 25-36. [[Paper]](http://aclweb.org/anthology/Q13-1003) [[Website]](http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos)

1. **Charades**: Sigurdsson, Gunnar A., et al. **Hollywood in homes: Crowdsourcing data collection for activity understanding.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1604.01753.pdf) [[Website]](https://allenai.org/plato/charades/)

1. **Charades-STA**: Gao, Jiyang, et al. **Tall: Temporal activity localization via language query.** arXiv preprint arXiv:1705.02101 (2017).[[Paper]](https://arxiv.org/pdf/1705.02101.pdf) [[Code]](https://github.com/jiyanggao/TALL)

1. **Distinct Describable Moments (DiDeMo)**: Hendricks, Lisa Anne, et al. **Localizing moments in video with natural language.** Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017. *Method name: MCN* [[Paper]](https://arxiv.org/pdf/1708.01641.pdf) [[Code]](https://github.com/LisaAnne/LocalizingMoments) [[Website]](https://people.eecs.berkeley.edu/~lisa_anne/didemo.html)

1. **ActivityNet Captions**: Krishna, Ranjay, et al. **Dense-captioning events in videos.** Proceedings of the IEEE International Conference on Computer Vision. 2017. [[Paper]](https://arxiv.org/pdf/1705.00754.pdf) [[Website]](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

1. **Charades-Ego**:  [[Website]](https://allenai.org/plato/charades/)
	- Sigurdsson, Gunnar, et al. **Actor and Observer: Joint Modeling of First and Third-Person Videos.** CVPR-IEEE Conference on Computer Vision & Pattern Recognition. 2018. [[Paper]](https://arxiv.org/pdf/1804.09627.pdf) [[Code]](https://github.com/gsig/actor-observer)
    - Sigurdsson, Gunnar A., et al. "Charades-Ego: A Large-Scale Dataset of Paired Third and First Person Videos." arXiv preprint arXiv:1804.09626 (2018). [[Paper]](https://arxiv.org/pdf/1804.09626.pdf) [[Code]](https://github.com/gsig/charades-algorithms)
    
1. **TEMPO**: Hendricks, Lisa Anne, et al. **Localizing Moments in Video with Temporal Language.** arXiv preprint arXiv:1809.01337 (2018). [[Paper]](https://arxiv.org/pdf/1809.01337.pdf) [[Code]](https://github.com/LisaAnne/TemporalLanguageRelease) [[Website]](https://people.eecs.berkeley.edu/~lisa_anne/tempo.html)

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

1. Margffoy-Tuay, Edgar, et al. **Dynamic multimodal instance segmentation guided by natural language queries.** Proceedings of the European Conference on Computer Vision (ECCV). 2018. [[Paper]](https://arxiv.org/pdf/1807.02257.pdf) [[Code]](https://github.com/BCV-Uniandes/DMS)

1. Plummer, Bryan A., et al. **Conditional image-text embedding networks.** Proceedings of the European Conference on Computer Vision (ECCV). 2018. [[Paper]](https://arxiv.org/pdf/1711.08389.pdf) [[Code]](https://github.com/BryanPlummer/cite)

1. Akbari, Hassan, et al. **Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding.** arXiv preprint arXiv:1811.11683 (2018). [[Paper]](https://arxiv.org/pdf/1811.11683v1.pdf) 

1. Kovvuri, Rama, and Ram Nevatia. **PIRC Net: Using Proposal Indexing, Relationships and Context for Phrase Grounding.** arXiv preprint arXiv:1812.03213 (2018). [[Paper]](https://arxiv.org/pdf/1812.03213v1.pdf)

1. Liu, Daqing, et al. **Explainability by Parsing: Neural Module Tree Networks for Natural Language Visual Grounding.** arXiv preprint arXiv:1812.03299 (2018). [[Paper]](https://arxiv.org/pdf/1812.03299v1.pdf)

1. Chen, Xinpeng, et al. **Real-Time Referring Expression Comprehension by Single-Stage Grounding Network.** arXiv preprint arXiv:1812.03426 (2018). [[Paper]](https://arxiv.org/pdf/1812.03426v1.pdf)

1. Wang, Peng, et al. **Neighbourhood Watch: Referring Expression Comprehension via Language-guided Graph Attention Networks.** arXiv preprint arXiv:1812.04794 (2018). [[Paper]](https://arxiv.org/pdf/1812.04794.pdf)

1. Deng, Chaorui, et al. **You Only Look & Listen Once: Towards Fast and Accurate Visual Grounding.** arXiv preprint arXiv:1902.04213 (2019). [[Paper]](https://arxiv.org/pdf/1902.04213.pdf)

### Natural Language Object Retrieval (Images)

1. Guadarrama, Sergio, et al. **Open-vocabulary Object Retrieval.** Robotics: science and systems. Vol. 2. No. 5. 2014. [[Paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.434.3000&rep=rep1&type=pdf) [[Code]](http://openvoc.berkeleyvision.org/)

1. Hu, Ronghang, et al. **Natural language object retrieval.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. *Method name: Spatial Context Recurrent
ConvNet (SCRC)* [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hu_Natural_Language_Object_CVPR_2016_paper.pdf) [[Code]](https://github.com/ronghanghu/natural-language-object-retrieval) [[Website]](http://ronghanghu.com/text_obj_retrieval/)

1. Wu, Fan et al. **An End-to-End Approach to Natural Language Object Retrieval via Context-Aware Deep Reinforcement Learning.** CoRR abs/1703.07579 (2017): n. pag. [[Paper]](https://arxiv.org/pdf/1703.07579.pdf) [[Code]](https://github.com/jxwufan/NLOR_A3C)

1. Li, Jianan, et al. **Deep attribute-preserving metric learning for natural language object retrieval.** Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017. [[Paper: ACM Link]](https://dl.acm.org/citation.cfm?id=3123439)

1. Nguyen, Anh, et al. **Object Captioning and Retrieval with Natural Language.** arXiv preprint arXiv:1803.06152 (2018). [[Paper]](https://arxiv.org/pdf/1803.06152.pdf) [[Website]](https://sites.google.com/site/objcaptioningretrieval/)

1. Plummer, Bryan A., et al. **Open-vocabulary Phrase Detection.** arXiv preprint arXiv:1811.07212 (2018). [[Paper]](https://arxiv.org/pdf/1811.07212.pdf) [[Code]](https://github.com/VisionLearningGroup/phrase-rcnn)

### Video Grounding (Activity Localization) using Natural Language:
1. Yu, Haonan, et al. **Grounded Language Learning from Video Described with Sentences** Proceedings of the Annual Meeting of the Association for Computational Linguistics. 2013. [[Paper]](https://www.aclweb.org/anthology/P13-1006)

1. Xu, Ran, et al. **Jointly Modeling Deep Video and Compositional Text to Bridge Vision and Language in a Unified Framework.** Proceedings of the AAAI Conference on Artificial Intelligence. 2015. [[Paper]](http://web.eecs.umich.edu/~jjcorso/pubs/xu_corso_AAAI2015_v2t.pdf)

1. Song, Young Chol, et al. **Unsupervised Alignment of Actions in Video with Text Descriptions** Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2016. [[Paper]](https://www.ijcai.org/Proceedings/16/Papers/289.pdf)

1. Gao, Jiyang, et al. **Tall: Temporal activity localization via language query.** arXiv preprint arXiv:1705.02101 (2017). *Method name: TALL* [[Paper]](https://arxiv.org/pdf/1705.02101.pdf) [[Code]](https://github.com/jiyanggao/TALL)

1. Hendricks, Lisa Anne, et al. **Localizing moments in video with natural language.** Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017. *Method name: MCN* [[Paper]](https://arxiv.org/pdf/1708.01641.pdf) [[Code]](https://github.com/LisaAnne/LocalizingMoments)

1. Khoreva, Anna, Anna Rohrbach, and Bernt Schiele. **Video Object Segmentation with Language Referring Expressions.** arXiv preprint arXiv:1803.08006 (2018). [[Paper]](https://arxiv.org/pdf/1803.08006.pdf) [[Website]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/video-segmentation/video-object-segmentation-with-language-referring-expressions/)

1. Xu, Huijuan, et al. **Text-to-Clip Video Retrieval with Early Fusion and Re-Captioning.** arXiv preprint arXiv:1804.05113 (2018). [[Paper]](https://arxiv.org/pdf/1804.05113.pdf) [[Code]](https://github.com/VisionLearningGroup/Text-to-Clip_Retrieval)

1. Liu, Bingbin, et al. **Temporal Modular Networks for Retrieving Complex Compositional Activities in Videos.** European Conference on Computer Vision. Springer, Cham, 2018. [[Paper]](http://svl.stanford.edu/assets/papers/liu2018eccv.pdf) [[Website]](https://clarabing.github.io/tmn/)

1. Liu, Meng, et al. **Attentive Moment Retrieval in Videos.** Proceedings of the International ACM SIGIR Conference . 2018. [[Paper]] (https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-video-retrieval.pdf) [[Website]] (https://sigir2018.wixsite.com/acrn)

1. Chen, Jingyuan, et al. **Temporally grounding natural sentence in video.** Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[Paper]](https://pdfs.semanticscholar.org/452a/ca244ef62a533d8b46a54c6212fe9fa3ce9a.pdf) 

1. Hendricks, Lisa Anne, et al. **Localizing Moments in Video with Temporal Language.** arXiv preprint arXiv:1809.01337 (2018). [[Paper]](https://arxiv.org/pdf/1809.01337.pdf) [[Code]](https://github.com/LisaAnne/TemporalLanguageRelease) [[Website]](https://people.eecs.berkeley.edu/~lisa_anne/tempo.html)

1. Zhang, Da, et al. **MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment.** arXiv preprint arXiv:1812.00087 (2018). [[Paper]](https://arxiv.org/pdf/1812.00087.pdf)

1. Ge, Runzhou, et al. **MAC: Mining Actiivity Concepts for Language-based Temporal Localization.** arXiv preprint arXiv:1811.08925 (2018). [[Paper]](https://arxiv.org/pdf/1811.08925.pdf) [[Code]](https://github.com/runzhouge/MAC)

1. Xu, Huijuan, et al. **Joint Event Detection and Description in Continuous Video Streams.** arXiv preprint arXiv:1802.10250 (2018). [[Paper]](https://arxiv.org/pdf/1802.10250.pdf) [[Code]](https://github.com/VisionLearningGroup/JEDDi-Net) 

1. He, Dongliang, et al. **Read, Watch, and Move: Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos.** Proceedings of the AAAI Conference on Artificial Intelligence. 2019. [[Paper]](https://arxiv.org/pdf/1901.06829.pdf)

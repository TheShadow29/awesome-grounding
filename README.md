# Awesome Visual Grounding
A curated list of research papers in grounding. Link to the code if available is also present.

Have a look at [SCOPE.md](SCOPE.md) to get familiar with what grounding means and the tasks considered in this repository. 

To maintaing the quality of the repo, I have gone through all the listed papers at least once before adding them to ensure their relevance to grounding. However, I might have missed some paper(s) or added some irrelevant paper(s). Feel free to open an issue in that case. I will go through the paper and then add / remove it. 

## Table of Contents
- [Contributing](#contributing)
- [Demos](#demos)
- [Other Compilations](#other-compilations)
- [Datasets](#datasets)
    - [Image Grounding Datasets](#image-grounding-datasets)
    - [Video Datasets](#video-datasets)
- [Paper Roadmap (Chronological Order):](#paper-roadmap-chronological-order)
	- [Visual Grounding / Referring Expressions (Images):](#visual-grounding--referring-expressions-images)
    - [Natural Language Object Retrieval (Images)](#natural-language-object-retrieval-images)
    - [Video Grounding (Activity Localization) using Natural Language:](#video-grounding-activity-localization-using-natural-language)
    - [Grounded Description (Image) (WIP)](#grounded-description-image-wip)
    - [Grounded Description (Video) (WIP)](#grounded-description-video-wip)
    - [Visual Grounding Pretraining](#visual-grounding-pretraining)

## Contributing
 Feel free to contact me via email (ark.sadhu2904@gmail.com) or open an issue or submit a pull request. To add a new paper via pull request:
 1. Fork the repo, change readme. Put the new paper under the correct heading, and place it at the correct chronological position.
 1. Copy its reference in MLA format
 1. Put ** around the title
 1. Provide link to the paper (arxiv/semantic scholar/conference proceedings).
 1. If code or website exists, link that too.
 1. Send a pull request. Ideally, I will review the request within a week.

## Demos
1. MATTNet demo: [http://vision2.cs.unc.edu/refer/comprehension](http://vision2.cs.unc.edu/refer/comprehension)

## Other Compilations:
Shoutout to some other awesome stuff on vision and language grounding:

1. Multi-modal Reading List by Paul Liang (@pliang279) : https://github.com/pliang279/awesome-multimodal-ml/
1. Temporal Grounding by Mu Ketong (@iworldtong): https://github.com/iworldtong/Awesome-Grounding-Natural-Language-in-Video
1. Temporal Grounding by WuJie (@WuJie1010): https://github.com/WuJie1010/Awesome-Temporally-Language-Grounding. Also, checkout their implementation of some of the popular papers: https://github.com/WuJie1010/Temporally-language-grounding

## Datasets
### Image Grounding Datasets

1. **Flickr30k**: Plummer, Bryan A., et al. **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** Proceedings of the IEEE international conference on computer vision. 2015. [[Paper]](https://arxiv.org/abs/1505.04870) [[Code]](https://github.com/BryanPlummer/pl-clc) [[Website]](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)

1. **RefClef**: Kazemzadeh, Sahar, et al. **Referitgame: Referring to objects in photographs of natural scenes.** Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014. [[Paper]](http://www.aclweb.org/anthology/D14-1086) [[Website]](http://tamaraberg.com/referitgame/)

1. **RefCOCOg**: Mao, Junhua, et al. **Generation and comprehension of unambiguous object descriptions.** Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [[Paper]](https://arxiv.org/pdf/1511.02283.pdf) [[Code]](https://github.com/mjhucla/Google_Refexp_toolbox)

1. **Visual Genome**: Krishna, Ranjay, et al. **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** International Journal of Computer Vision 123.1 (2017): 32-73. [[Paper]](https://arxiv.org/pdf/1602.07332.pdf) [[Website]](https://visualgenome.org/)

1. **RefCOCO and RefCOCO+**: 1. Yu, Licheng, et al. **Modeling context in referring expressions.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1608.00272.pdf)[[Code]](https://github.com/lichengunc/refer)

1. **GuessWhat**: De Vries, Harm, et al. **Guesswhat?! visual object discovery through multi-modal dialogue.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017. [[Paper]](https://arxiv.org/abs/1611.08481) [[Code]](https://github.com/GuessWhatGame/guesswhat/) [[Website]](https://guesswhat.ai/#) 

1. **Clevr-ref+**: Liu, Runtao, et al. **Clevr-ref+: Diagnosing visual reasoning with referring expressions.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019. [[Paper]](https://arxiv.org/pdf/1901.00850.pdf) [[Code]](https://github.com/ccvl/clevr-refplus-dataset-gen) [[Website]](https://cs.jhu.edu/~cxliu/2019/clevr-ref+)

1. **KB-Ref**: Wang, Peng, et al. **Give Me Something to Eat: Referring Expression Comprehension with Commonsense Knowledge.** Proceedings of the 28th ACM International Conference on Multimedia. 2020. [[Paper]](https://arxiv.org/pdf/2006.01629) [[Code]](https://github.com/wangpengnorman/KB-Ref_dataset)

1. **Ref-Reasoning**: Yang, Sibei, Guanbin Li, and Yizhou Yu. **Graph-structured referring expression reasoning in the wild.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Graph-Structured_Referring_Expression_Reasoning_in_the_Wild_CVPR_2020_paper.pdf) [[Code]](https://github.com/sibeiyang/sgmn) [[Website]](https://sibeiyang.github.io/dataset/ref-reasoning/)

1. **Cops-Ref**: Chen, Zhenfang, et al. **Cops-Ref: A new Dataset and Task on Compositional Referring Expression Comprehension.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Cops-Ref_A_New_Dataset_and_Task_on_Compositional_Referring_Expression_CVPR_2020_paper.pdf) [[Code]](https://github.com/zfchenUnique/Cops-Ref)

1. **SUNRefer**: Liu, Haolin, et al. **Refer-it-in-RGBD: A Bottom-up Approach for 3D Visual Grounding in RGBD Images.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[Paper]](https://arxiv.org/pdf/2103.07894) [[Code]](https://github.com/UncleMEDM/Refer-it-in-RGBD) [[Website]](https://unclemedm.github.io/Refer-it-in-RGBD/)



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

1. **ActivityNet-Entities**: Zhou, Luowei, et al. **Grounded video description.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019. [[Paper]](https://arxiv.org/pdf/1812.06587.pdf) [[Code]](https://github.com/facebookresearch/ActivityNet-Entities) 


### Embodied Agents Platforms:

1. **Matterport3D**: Chang, Angel, et al. **Matterport3d: Learning from rgb-d data in indoor environments.** arXiv preprint arXiv:1709.06158 (2017). [[Paper]](https://arxiv.org/pdf/1711.07280.pdf) [[Code]](https://github.com/niessner/Matterport) [[Website]](https://github.com/niessner/Matterport)
	- Photorealistic rooms

1. **AI2-THOR**: Kolve, Eric, et al. **Ai2-thor: An interactive 3d environment for visual ai.** arXiv preprint arXiv:1712.05474 (2017). [[Paper]](https://arxiv.org/pdf/1712.05474.pdf) [[Website]](https://ai2thor.allenai.org/)
	- Actionable objects!

1. **Habitat AI**: Savva, Manolis, et al. **Habitat: A platform for embodied ai research.** Proceedings of the IEEE International Conference on Computer Vision. 2019. (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Savva_Habitat_A_Platform_for_Embodied_AI_Research_ICCV_2019_paper.pdf) [[Website]](https://aihabitat.org/)

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

1. Li, Ruiyu, et al. **Referring image segmentation via recurrent refinement networks.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.[[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Referring_Image_Segmentation_CVPR_2018_paper.pdf) [[Code]](https://github.com/liruiyu/referseg_rrn)

1. Zhang, Yundong, Juan Carlos Niebles, and Alvaro Soto. **Interpretable Visual Question Answering by Visual Grounding from Attention Supervision Mining.** arXiv preprint arXiv:1808.00265 (2018). [[Paper]](https://arxiv.org/pdf/1808.00265.pdf) 

1. Chen, Kan, Jiyang Gao, and Ram Nevatia. **Knowledge aided consistency for weakly supervised phrase grounding.** arXiv preprint arXiv:1803.03879 (2018). [[Paper]](https://arxiv.org/abs/1803.03879) [[Code]](https://github.com/kanchen-usc/KAC-Net)

1. Zhang, Hanwang, Yulei Niu, and Shih-Fu Chang. **Grounding referring expressions in images by variational context.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Grounding_Referring_Expressions_CVPR_2018_paper.pdf) [[Code]](https://github.com/yuleiniu/vc/)

1. Cirik, Volkan, Taylor Berg-Kirkpatrick, and Louis-Philippe Morency. **Using syntax to ground referring expressions in natural images.** arXiv preprint arXiv:1805.10547 (2018).[[Paper]](https://arxiv.org/pdf/1805.10547.pdf) [[Code]](https://github.com/volkancirik/groundnet)

1. Margffoy-Tuay, Edgar, et al. **Dynamic multimodal instance segmentation guided by natural language queries.** Proceedings of the European Conference on Computer Vision (ECCV). 2018. [[Paper]](https://arxiv.org/pdf/1807.02257.pdf) [[Code]](https://github.com/BCV-Uniandes/DMS)

1. Shi, Hengcan, et al. **Key-word-aware network for referring expression image segmentation.** Proceedings of the European Conference on Computer Vision (ECCV). 2018.[[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengcan_Shi_Key-Word-Aware_Network_for_ECCV_2018_paper.pdf) [[Code]](https://github.com/shihengcan/key-word-aware-network-pycaffe)

1. Plummer, Bryan A., et al. **Conditional image-text embedding networks.** Proceedings of the European Conference on Computer Vision (ECCV). 2018. [[Paper]](https://arxiv.org/pdf/1711.08389.pdf) [[Code]](https://github.com/BryanPlummer/cite)

1. Akbari, Hassan, et al. **Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding.** arXiv preprint arXiv:1811.11683 (2018). [[Paper]](https://arxiv.org/pdf/1811.11683v1.pdf) 

1. Kovvuri, Rama, and Ram Nevatia. **PIRC Net: Using Proposal Indexing, Relationships and Context for Phrase Grounding.** arXiv preprint arXiv:1812.03213 (2018). [[Paper]](https://arxiv.org/pdf/1812.03213v1.pdf)

1. Chen, Xinpeng, et al. **Real-Time Referring Expression Comprehension by Single-Stage Grounding Network.** arXiv preprint arXiv:1812.03426 (2018). [[Paper]](https://arxiv.org/pdf/1812.03426v1.pdf)

1. Wang, Peng, et al. **Neighbourhood Watch: Referring Expression Comprehension via Language-guided Graph Attention Networks.** arXiv preprint arXiv:1812.04794 (2018). [[Paper]](https://arxiv.org/pdf/1812.04794.pdf)

1. Liu, Daqing, et al. **Learning to Assemble Neural Module Tree Networks for Visual Grounding.** Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2019. [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Learning_to_Assemble_Neural_Module_Tree_Networks_for_Visual_Grounding_ICCV_2019_paper.pdf) [[Code]](https://github.com/daqingliu/NMTree)

1. **RETRACTED (see [#2](https://github.com/TheShadow29/awesome-grounding/pull/2))**:  Deng, Chaorui, et al. **You Only Look & Listen Once: Towards Fast and Accurate Visual Grounding.** arXiv preprint arXiv:1902.04213 (2019). [[Paper]](https://arxiv.org/pdf/1902.04213.pdf)

1. Hong, Richang, et al. **Learning to Compose and Reason with Language Tree Structures for Visual Grounding.** IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI). 2019. [[Paper]](https://arxiv.org/pdf/1906.01784.pdf)

1. Liu, Xihui, et al. **Improving Referring Expression Grounding with Cross-modal Attention-guided Erasing.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019. [[Paper]](https://arxiv.org/pdf/1903.00839.pdf) 

1. Dogan, Pelin, Leonid Sigal, and Markus Gross. **Neural Sequential Phrase Grounding (SeqGROUND).** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. (CVPR) 2019. [[Paper]](https://arxiv.org/pdf/1903.07669.pdf)

1. Datta, Samyak, et al. **Align2ground: Weakly supervised phrase grounding guided by image-caption alignment.** arXiv preprint arXiv:1903.11649 (2019). (ICCV 2019) [[Paper]](https://arxiv.org/pdf/1903.11649.pdf)

1. Fang, Zhiyuan, et al. **Modularized textual grounding for counterfactual resilience.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. (CVPR) 2019. [[Paper]](https://arxiv.org/pdf/1904.03589.pdf)

1. Ye, Linwei, et al. **Cross-Modal Self-Attention Network for Referring Image Segmentation.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. (CVPR) 2019. [[Paper]](https://arxiv.org/pdf/1904.04745.pdf)

1. Yang, Sibei, Guanbin Li, and Yizhou Yu. **Cross-Modal Relationship Inference for Grounding Referring Expressions.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. (CVPR) 2019. [[Paper]](https://arxiv.org/pdf/1906.04464.pdf)

1. Yang, Sibei, Guanbin Li, and Yizhou Yu. **Dynamic Graph Attention for Referring Expression Comprehension.** arXiv preprint arXiv:1909.08164 (2019). (ICCV 2019) [[Paper]](https://arxiv.org/pdf/1909.08164.pdf) [[Code]](https://github.com/sibeiyang/sgmn/tree/master/lib/dga_models)

1. Wang, Josiah, and Lucia Specia. **Phrase Localization Without Paired Training Examples.** arXiv preprint arXiv:1908.07553 (2019). (ICCV 2019) [[Paper]](https://arxiv.org/abs/1908.07553) [[Code]](https://github.com/josiahwang/phraseloceval) 

1. Yang, Zhengyuan, et al. **A Fast and Accurate One-Stage Approach to Visual Grounding.** arXiv preprint arXiv:1908.06354 (2019). (ICCV 2019) [[Paper]](https://arxiv.org/pdf/1908.06354.pdf) [[Code]](https://github.com/zyang-ur/onestage_grounding)

1. Sadhu, Arka, Kan Chen, and Ram Nevatia. **Zero-Shot Grounding of Objects from Natural Language Queries.** arXiv preprint arXiv:1908.07129 (2019).(ICCV 2019) [[Paper]](https://arxiv.org/abs/1908.07129) [[Code]](https://github.com/TheShadow29/zsgnet-pytorch)
*Disclaimer: I am an author of the paper*

1. Liu, Xuejing, et al. **Adaptive Reconstruction Network for Weakly Supervised Referring Expression Grounding.** arXiv preprint arXiv:1908.10568 (2019). (ICCV 2019) [[Paper]](https://arxiv.org/pdf/1908.10568.pdf) [[Code]](https://github.com/GingL/ARN)

1. Chen, Yi-Wen, et al. **Referring Expression Object Segmentation with Caption-Aware Consistency.** arXiv preprint arXiv:1910.04748 (2019). (BMVC 2019) [[Paper]](https://arxiv.org/abs/1910.04748) [[Code]](https://github.com/wenz116/lang2seg)

1. Liu, Jiacheng, and Julia Hockenmaier. **Phrase Grounding by Soft-Label Chain Conditional Random Field.** arXiv preprint arXiv:1909.00301 (2019) (EMNLP 2019). [[Paper]](https://arxiv.org/pdf/1909.00301.pdf) [[Code]](https://github.com/liujch1998/SoftLabelCCRF)

1. Liu, Yongfei, Wan Bo, Zhu Xiaodan and He Xuming. **Learning Cross-modal Context Graph for Visual Grounding.** arXiv preprint arXiv: (2019) (AAAI-2020). [[Paper]](https://arxiv.org/pdf/1911.09042.pdf) [[Code]](https://github.com/youngfly11/LCMCG-PyTorch)

1. Yu, Tianyu, et al. **Cross-Modal Omni Interaction Modeling for Phrase Grounding.** Proceedings of the 28th ACM International Conference on Multimedia. ACM 2020. [[Paper: ACM Link]](https://dl.acm.org/doi/abs/10.1145/3394171.3413846) [[Code]](https://github.com/yiranyyu/Phrase-Grounding)

1. Qiu, Heqian, et al. **Language-Aware Fine-Grained Object Representation for Referring Expression Comprehension.** Proceedings of the 28th ACM International Conference on Multimedia. ACM 2020. [[Paper: ACM Link]](https://dl.acm.org/doi/abs/10.1145/3394171.3413850)

1. Wang, Qinxin, et al. **MAF: Multimodal Alignment Framework for Weakly-Supervised Phrase Grounding.** arXiv preprint arXiv:2010.05379 (2020). [[Paper]](https://arxiv.org/pdf/2010.05379) [[Code]](https://github.com/qinzzz/Multimodal-Alignment-Framework)

1. Liao, Yue, et al. **A real-time cross-modality correlation filtering method for referring expression comprehension.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liao_A_Real-Time_Cross-Modality_Correlation_Filtering_Method_for_Referring_Expression_Comprehension_CVPR_2020_paper.pdf)

1. Hu, Zhiwei, et al. **Bi-directional relationship inferring network for referring image segmentation.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Bi-Directional_Relationship_Inferring_Network_for_Referring_Image_Segmentation_CVPR_2020_paper.pdf) [[Code]](https://github.com/fengguang94/CVPR2020-BRINet)

1. Yang, Sibei, Guanbin Li, and Yizhou Yu. **Graph-structured referring expression reasoning in the wild.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Graph-Structured_Referring_Expression_Reasoning_in_the_Wild_CVPR_2020_paper.pdf) [[Code]](https://github.com/sibeiyang/sgmn)

1. Luo, Gen, et al. **Multi-task collaborative network for joint referring expression comprehension and segmentation.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Multi-Task_Collaborative_Network_for_Joint_Referring_Expression_Comprehension_and_Segmentation_CVPR_2020_paper.pdf) [[Code]](https://github.com/luogen1996/MCN)

1. Gupta, Tanmay, et al. **Contrastive learning for weakly supervised phrase grounding.** Proceedings of the European Conference on Computer Vision (ECCV). 2020. [[Paper]](https://arxiv.org/pdf/2006.09920) [[Code]](https://github.com/BigRedT/info-ground)

1. Yang, Zhengyuan, et al. **Improving one-stage visual grounding by recursive sub-query construction.** Proceedings of the European Conference on Computer Vision (ECCV). 2020. [[Paper]](https://arxiv.org/pdf/2008.01059) [[Code]](https://github.com/zyang-ur/ReSC)

1. Wang, Liwei, et al. **Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[Paper]](https://arxiv.org/pdf/2007.01951)

1. Sun, Mingjie, Jimin Xiao, and Eng Gee Lim. **Iterative Shrinking for Referring Expression Grounding Using Deep Reinforcement Learning.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[Paper]](https://arxiv.org/pdf/2007.01951) [[Code]](https://github.com/insomnia94/ISREG)

1. Liu, Haolin, et al. **Refer-it-in-RGBD: A Bottom-up Approach for 3D Visual Grounding in RGBD Images.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[Paper]](https://arxiv.org/pdf/2103.07894) [[Code]](https://github.com/UncleMEDM/Refer-it-in-RGBD)

1. Liu, Yongfei, et al. **Relation-aware Instance Refinement for Weakly Supervised Visual Grounding.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[Paper]](https://arxiv.org/pdf/2103.12989) [[Code]](https://github.com/youngfly11/ReIR-WeaklyGrounding.pytorch)

1. Lin, Xiangru, Guanbin Li, and Yizhou Yu. **Scene-Intuitive Agent for Remote Embodied Visual Grounding.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[Paper]](https://arxiv.org/pdf/2103.12944)

1. Sun, Mingjie, et al. **Discriminative triad matching and reconstruction for weakly referring expression grounding.** IEEE transactions on pattern analysis and machine intelligence (TPAMI 2021). [[Paper]](https://livrepository.liverpool.ac.uk/3116000/1/manuscript.pdf) [[Code]](https://github.com/insomnia94/DTWREG)

1. Mu, Zongshen, et al. **Disentangled Motif-aware Graph Learning for Phrase Grounding.** arXiv preprint arXiv:2104.06008 (AAAI 2021). [[Paper]](https://www.aaai.org/AAAI21Papers/AAAI-2589.MuZ.pdf)

1. Chen, Long, et al. **Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding.** arXiv preprint arXiv:2009.01449 (AAAI-2021). [[Paper]](https://arxiv.org/pdf/2009.01449) [[Code]](https://github.com/ChopinSharp/ref-nms)

1. Deng, Jiajun, et al. **TransVG: End-to-End Visual Grounding with Transformers.** arXiv preprint arXiv:2104.08541 (2021). [[Paper]](https://arxiv.org/pdf/2104.08541) [[Unofficial Code]](https://github.com/nku-shengzheliu/Pytorch-TransVG)

1. Du, Ye, et al. **Visual Grounding with Transformers.** arXiv preprint arXiv:2105.04281 (2021). [[Paper]](https://arxiv.org/pdf/2105.04281)

1. Kamath, Aishwarya, et al. **MDETR--Modulated Detection for End-to-End Multi-Modal Understanding.** arXiv preprint arXiv:2104.12763 (2021). [[Paper]](https://arxiv.org/pdf/2104.12763)

### Natural Language Object Retrieval (Images)

1. Guadarrama, Sergio, et al. **Open-vocabulary Object Retrieval.** Robotics: science and systems. Vol. 2. No. 5. 2014. [[Paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.434.3000&rep=rep1&type=pdf) [[Code]](http://openvoc.berkeleyvision.org/)

1. Hu, Ronghang, et al. **Natural language object retrieval.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. *Method name: Spatial Context Recurrent
ConvNet (SCRC)* [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hu_Natural_Language_Object_CVPR_2016_paper.pdf) [[Code]](https://github.com/ronghanghu/natural-language-object-retrieval) [[Website]](http://ronghanghu.com/text_obj_retrieval/)

1. Wu, Fan et al. **An End-to-End Approach to Natural Language Object Retrieval via Context-Aware Deep Reinforcement Learning.** CoRR abs/1703.07579 (2017): n. pag. [[Paper]](https://arxiv.org/pdf/1703.07579.pdf) [[Code]](https://github.com/jxwufan/NLOR_A3C)

1. Li, Jianan, et al. **Deep attribute-preserving metric learning for natural language object retrieval.** Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017. [[Paper: ACM Link]](https://dl.acm.org/citation.cfm?id=3123439)

1. Nguyen, Anh, et al. **Object Captioning and Retrieval with Natural Language.** arXiv preprint arXiv:1803.06152 (2018). [[Paper]](https://arxiv.org/pdf/1803.06152.pdf) [[Website]](https://sites.google.com/site/objcaptioningretrieval/)

1. Plummer, Bryan A., et al. **Open-vocabulary Phrase Detection.** arXiv preprint arXiv:1811.07212 (2018). [[Paper]](https://arxiv.org/pdf/1811.07212.pdf) [[Code]](https://github.com/VisionLearningGroup/phrase-rcnn)

### Grounding Relations / Referring Relations

1. Krishna, Ranjay, et al. **Referring relationships.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018. [[Paper]](https://arxiv.org/pdf/1803.10362.pdf) [[Code]](https://github.com/StanfordVL/ReferringRelationships) [[Website]](https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/index.html)

1. Raboh, Moshiko et al. **Differentiable Scene Graphs.** (2019). [[Paper]](https://arxiv.org/pdf/1902.10200.pdf)

1. Conser, Erik, et al. **Revisiting Visual Grounding.** arXiv preprint arXiv:1904.02225 (2019).
[[Paper]](https://arxiv.org/pdf/1904.02225.pdf)
	- Critique of Referring Relationship paper

### Video Grounding (Activity Localization) using Natural Language:
1. Yu, Haonan, et al. **Grounded Language Learning from Video Described with Sentences** Proceedings of the Annual Meeting of the Association for Computational Linguistics. 2013. [[Paper]](https://www.aclweb.org/anthology/P13-1006)

1. Xu, Ran, et al. **Jointly Modeling Deep Video and Compositional Text to Bridge Vision and Language in a Unified Framework.** Proceedings of the AAAI Conference on Artificial Intelligence. 2015. [[Paper]](http://web.eecs.umich.edu/~jjcorso/pubs/xu_corso_AAAI2015_v2t.pdf)

1. Song, Young Chol, et al. **Unsupervised Alignment of Actions in Video with Text Descriptions** Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2016. [[Paper]](https://www.ijcai.org/Proceedings/16/Papers/289.pdf)

1. Gao, Jiyang, et al. **Tall: Temporal activity localization via language query.** arXiv preprint arXiv:1705.02101 (2017). *Method name: TALL* [[Paper]](https://arxiv.org/pdf/1705.02101.pdf) [[Code]](https://github.com/jiyanggao/TALL)

1. Hendricks, Lisa Anne, et al. **Localizing moments in video with natural language.** Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017. *Method name: MCN* [[Paper]](https://arxiv.org/pdf/1708.01641.pdf) [[Code]](https://github.com/LisaAnne/LocalizingMoments)

1. Khoreva, Anna, Anna Rohrbach, and Bernt Schiele. **Video Object Segmentation with Language Referring Expressions.** arXiv preprint arXiv:1803.08006 (2018). [[Paper]](https://arxiv.org/pdf/1803.08006.pdf) [[Website]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/video-segmentation/video-object-segmentation-with-language-referring-expressions/)

1. Xu, Huijuan, et al. **Joint Event Detection and Description in Continuous Video Streams.** arXiv preprint arXiv:1802.10250 (2018). [[Paper]](https://arxiv.org/pdf/1802.10250.pdf) [[Code]](https://github.com/VisionLearningGroup/JEDDi-Net) 

1. Xu, Huijuan, et al. **Text-to-Clip Video Retrieval with Early Fusion and Re-Captioning.** arXiv preprint arXiv:1804.05113 (2018). [[Paper]](https://arxiv.org/pdf/1804.05113.pdf) [[Code]](https://github.com/VisionLearningGroup/Text-to-Clip_Retrieval)

1. Liu, Bingbin, et al. **Temporal Modular Networks for Retrieving Complex Compositional Activities in Videos.** European Conference on Computer Vision. Springer, Cham, 2018. [[Paper]](http://svl.stanford.edu/assets/papers/liu2018eccv.pdf) [[Website]](https://clarabing.github.io/tmn/)

1. Liu, Meng, et al. **Attentive Moment Retrieval in Videos.** Proceedings of the International ACM SIGIR Conference . 2018. [[Paper]](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir18-video-retrieval.pdf) [[Website]](https://sigir2018.wixsite.com/acrn)

1. Chen, Jingyuan, et al. **Temporally grounding natural sentence in video.** Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[Paper]](https://pdfs.semanticscholar.org/452a/ca244ef62a533d8b46a54c6212fe9fa3ce9a.pdf) 

1. Hendricks, Lisa Anne, et al. **Localizing Moments in Video with Temporal Language.** arXiv preprint arXiv:1809.01337 (2018). [[Paper]](https://arxiv.org/pdf/1809.01337.pdf) [[Code]](https://github.com/LisaAnne/TemporalLanguageRelease) [[Website]](https://people.eecs.berkeley.edu/~lisa_anne/tempo.html)

1. Wu, Aming, and Yahong Han. **Multi-modal Circulant Fusion for Video-to-Language and Backward.** IJCAI. Vol. 3. No. 4. 2018. [[Paper]](https://www.ijcai.org/Proceedings/2018/0143.pdf) [[Code]](https://github.com/AmingWu/Multi-modal-Circulant-Fusion)

1. Ge, Runzhou, et al. **MAC: Mining Actiivity Concepts for Language-based Temporal Localization.** arXiv preprint arXiv:1811.08925 (2018). [[Paper]](https://arxiv.org/pdf/1811.08925.pdf) [[Code]](https://github.com/runzhouge/MAC)

1. Zhang, Da, et al. **MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment.** arXiv preprint arXiv:1812.00087 (2018). [[Paper]](https://arxiv.org/pdf/1812.00087.pdf)

1. He, Dongliang, et al. **Read, Watch, and Move: Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos.** Proceedings of the AAAI Conference on Artificial Intelligence. 2019. [[Paper]](https://arxiv.org/pdf/1901.06829.pdf)

1. Wang, Weining, Yan Huang, and Liang Wang. **Language-Driven Temporal Activity Localization: A Semantic Matching Reinforcement Learning Model.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Language-Driven_Temporal_Activity_Localization_A_Semantic_Matching_Reinforcement_Learning_Model_CVPR_2019_paper.pdf)

1. Ghosh, Soham, et al. **ExCL: Extractive Clip Localization Using Natural Language Descriptions.** arXiv preprint arXiv:1904.02755 (2019). [[Paper]](https://www.aclweb.org/anthology/N19-1198)

1. Chen, Shaoxiang, and Yu-Gang Jiang. **Semantic Proposal For Activity Localization In Videos Via Sentence Query.** Proceedings of the AAAI Conference on Artificial Intelligence. 2019.[[Paper]](http://yugangjiang.info/publication/19AAAI-actionlocalization.pdf)

1. Yuan Y, Mei T, Zhu W. **To Find Where You Talk: Temporal Sentence Localization In Video With Attention Based Location Regression.** Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 9159-9166. [[Paper]](https://arxiv.org/pdf/1804.07014.pdf)

1. Mithun N C, Paul S, Roy-Chowdhury A K. **Weakly Supervised Video Moment Retrieval From Text Queries**. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 11592-11601.[[Paper]](https://arxiv.org/pdf/1904.03282.pdf)

1. Escorcia, Victor, et al. **Temporal Localization of Moments in Video Collections with Natural Language.** arXiv preprint arXiv:1907.12763 (2019). (ICCV 2019) [[Paper]](https://arxiv.org/pdf/1907.12763.pdf) [[Code]](https://github.com/escorciav/moments-retrieval-page)

1. Wang, Jingwen, Lin Ma, and Wenhao Jiang. **Temporally Grounding Language Queries in Videos by Contextual Boundary-aware Prediction.** arXiv preprint arXiv:1909.05010 (2019). (AAAI 2020) [[Paper]](https://arxiv.org/pdf/1909.05010.pdf) [[Code]](https://github.com/JaywongWang/CBP)


### Grounded Description (Image) (WIP)
1. Hendricks, Lisa Anne, et al. **Generating visual explanations.** European Conference on Computer Vision. Springer, Cham, 2016. [[Paper]](https://arxiv.org/pdf/1603.08507.pdf) [[Code]](https://github.com/LisaAnne/ECCV2016/) [[Pytorch Code]](https://github.com/salaniz/pytorch-gve-lrcn)

1. Jiang, Ming, et al. **TIGEr: Text-to-Image Grounding for Image Caption Evaluation.** arXiv preprint arXiv:1909.02050 (2019). (EMNLP 2019) [[Paper]](https://arxiv.org/pdf/1909.02050.pdf) [[Code]](https://github.com/SeleenaJM/CapEval)

1. Lee, Jason, Kyunghyun Cho, and Douwe Kiela. **Countering language drift via visual grounding.** arXiv preprint arXiv:1909.04499 (2019). (EMNLP 2019) [[Paper]](https://arxiv.org/pdf/1909.04499.pdf)

### Grounded Description (Video) (WIP)

1. Ma, Chih-Yao, et al. **Grounded Objects and Interactions for Video Captioning.** arXiv preprint arXiv:1711.06354 (2017). [[Paper]](https://arxiv.org/pdf/1711.06354.pdf)

1. Zhou, Luowei, et al. **Grounded video description.** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019. [[Paper]](https://arxiv.org/abs/1812.06587) [[Code]](https://github.com/facebookresearch/grounded-video-description)

### Visual Grounding Pretraining

1. Sun, Chen, et al. **Videobert: A joint model for video and language representation learning.** arXiv preprint arXiv:1904.01766 (2019). [[Paper]](https://arxiv.org/abs/1904.01766)

1. Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee. **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.** arXiv preprint arXiv:1908.02265 (Neurips 2019) [[Paper]](https://arxiv.org/pdf/1908.02265.pdf) [[Code]](https://github.com/jiasenlu/vilbert_beta)

1. Li, Liunian Harold, et al. **VisualBERT: A Simple and Performant Baseline for Vision and Language.** arXiv preprint arXiv:1908.03557 (2019). [[Paper]](https://arxiv.org/pdf/1908.03557.pdf) [[Code]](https://github.com/uclanlp/visualbert)

1. Li, Gen, et al. **Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training.** arXiv preprint arXiv:1908.06066 (2019). [[Paper]](https://arxiv.org/pdf/1908.06066.pdf)

1. Tan, Hao, and Mohit Bansal. **Lxmert: Learning cross-modality encoder representations from transformers.** arXiv preprint arXiv:1908.07490 (2019). [[Paper]](https://arxiv.org/pdf/1908.07490.pdf) [[Code]](https://github.com/airsplay/lxmert)

1. Su, Weijie, et al. **Vl-bert: Pre-training of generic visual-linguistic representations.** arXiv preprint arXiv:1908.08530 (2019). [[Paper]](https://arxiv.org/pdf/1908.08530.pdf)

1. Chen, Yen-Chun, et al. **UNITER: Learning UNiversal Image-TExt Representations.** arXiv preprint arXiv:1909.11740 (2019). [[Paper]](https://arxiv.org/pdf/1909.11740.pdf)

### Grounding for Embodied Agents (WIP):

1. Shridhar, Mohit, et al. **ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks.** arXiv preprint arXiv:1912.01734 (2019). [[Paper]](https://arxiv.org/pdf/1912.01734.pdf) [[Code]](https://github.com/askforalfred/alfred) [[Website]](https://askforalfred.com/) 

### Misc:
1. Han, Xudong, Philip Schulz, and Trevor Cohn. **Grounding learning of modifier dynamics: An application to color naming.** arXiv preprint arXiv:1909.07586 (2019). (EMNLP 2019) [[Paper]](https://arxiv.org/pdf/1909.07586.pdf) [[Code]](https://github.com/HanXudong/GLoM)

1. Yu, Xintong, et al. **What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues.** arXiv preprint arXiv:1909.00421 (2019). (EMNLP 2019) [[Paper]](https://arxiv.org/pdf/1909.00421.pdf) [[Code]](https://github.com/HKUST-KnowComp/Visual_PCR)

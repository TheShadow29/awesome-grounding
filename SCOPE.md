# Grounding Scope

Here I outline what grounding entails and the topics considered in this repository.

~~Visual grounding task refers to localizing an object given a query or a sentence. It is also sometimes called referring expression comprehension. Referring expression is basically uniquely identifying the object in question. I have not included papers which do only referring expression generation, however if they also do the comprehension (or only comprehension) they have been included.~~

The above definition of Visual grounding is a bit restrictive. In general, grounding X in Y (here X and Y are two different domains) would involve associating phrases (or a word) in X to phrases (or a word) in Y. This new definition makes "grounding" as a problem fundamental to solving multi-modal tasks. In this repository, I focus only on the Vision-Language grounding tasks.

Following this definition, here is the list of tasks this repository would cover:
- Phrase Grounding in Images aka Referring Expressions: Input is a Query Phrase and an Image, output is the referred bounding box. 
- Temporal Grounding in Videos: Input is a Query Phrase and a Video (30s-60s long), output is the timestamps to find where the queried action was performed.
- Grounded VQA: Standard VQA task, but the answer is deemed correct only if the correct objects are localized. (TODO). Also note for VQA you can also use this repo as a guide [https://github.com/JamesChuanggg/awesome-vqa](https://github.com/JamesChuanggg/awesome-vqa).
- Grounded Description: Input is an Image or a Video, the output is a caption describing the visual media. The caption must further be grounded in the visual media, i.e., provides evidence for the generated caption and ensure that there is no hallucination. (TODO)
- Grounded Expression Generation: Input is an Image and a bounding box around a particular object, the output is a natural language phrase which describes the object so that it is uniquely identifiable in the image. (TODO).
- Grounding Relations: Input is a three-tuple consisting of subject-verb-object, the task is to localize the subject and the object in the image. (TODO)
- Grounding for Embodied Agents, Visual Navigation: Input is a Query Phrase, usually a command or a question, and an Agent which can traverse in a simulated environment. (TODO)

I will likely add more tasks as I believe there will soon be many new works based on grounding. 

Note that a primary criteria for inclusion would be wheather the evaluation considers grounding or not. For instance, if there is a paper on VQA using attention mechanisms which inherently perform some kind of associations but doesn't provide an evaluation of grounding the objects then it would not be included. If instead they do provide a metric related to grounding, like considering the answer correct only when the grounded object is correct, then it would be added. Similary, an image captioning paper (even if it were to use attention) would not be included, unless it contains evaluations where the evidence is considered like in grounded descriptions.


If you believe there are grounding tasks which are not covered here, feel free to create an issue and/or send me a mail (ark.sadhu2904@gmail.com).

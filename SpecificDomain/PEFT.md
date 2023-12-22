## LoRA的基本原理和概念

LoRA（Low-Rank Adaptation）是一种用于大型语言模型的方法，它冻结预训练模型的权重，并在Transformer架构的每一层注入可训练的秩分解矩阵，大大减少了下游任务的可训练参数数量[1]。例如，使用GPT-3 175B作为例子，部署独立实例的微调模型，每个模型都有175B的参数，这是极其昂贵的。LoRA可以将可训练参数的数量减少10000倍，GPU内存需求减少3倍[1]。

## LoRA的应用和优势

LoRA在RoBERTa，DeBERTa，GPT-2和GPT-3的模型质量上表现得与微调相当或更好，尽管它的可训练参数更少，训练吞吐量更高，与适配器不同，没有额外的推理延迟[1]。LoRA的优点在于其低秩分解直观，在许多场景下与全量微调的效果一致，以及在预测阶段可以直接将增量合并成单个矩阵，从而不增加推理成本[4]。

## LoRA的训练方法和技术细节

LoRA的训练方法是将预训练的参数矩阵进行低秩分解，而不是直接微调这些参数。具体来说，对于预训练的参数矩阵，LoRA不直接微调，而是对增量做低秩分解假设。其中，或者之一用全零初始化，固定不变，优化器只优化。由于本征维度很小的结论，所以我们可以取得很小，很多时候我们甚至可以直接取1[4]。

在训练过程中，求模型梯度是主要的计算量，如果全量更新，那么所用的梯度是，而LoRA所用的梯度则是和，它们是建立在全量更新的梯度基础上的，所以理论上LoRA的计算量比全量更新还大。然而，实际使用时LoRA的训练速度也会变快，原因包括只更新了部分参数，减少了通信时间，以及采用了各种低精度加速技术，如FP16、FP8或者INT8量化等[4]。

Citations:
[1] https://arxiv.org/abs/2106.09685
[2] https://www.ebyte.com/new-view-info.aspx?id=2337
[3] https://www.tuya.com/cn/industry-details/Kb0lbmbpqg2sv
[4] https://www.sohu.com/a/668523947_121119001
[5] https://iot-book.github.io/9_LoRa/S2_LoRa%E9%80%9A%E4%BF%A1%E5%AE%9E%E9%AA%8C/
[6] https://www.semtech.cn/lora/why-lora
[7] https://blog.csdn.net/weixin_44826203/article/details/129733930
[8] https://openreview.net/forum?id=nZeVKeeFYf9
[9] https://www.techphant.cn/blog/7986.html
[10] https://www.techphant.cn/blog/3159.html
[11] https://www.cas.cn/syky/202109/t20210930_4807920.shtml
[12] https://blog.csdn.net/HiWangWenBing/article/details/109550068
[13] https://www.szrfstar.com/news/%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82%E5%9F%BA%E4%BA%8ELoRa%E6%8A%80%E6%9C%AF%E7%9A%84LoRaMESH%E5%BA%94%E7%94%A8%E4%BC%98%E5%8A%BF-cn.html
[14] http://nick.txtcc.com/index.php/ai/2509
[15] https://youtube.com/watch?v=PXWYUTMt-AU
[16] https://www.techphant.cn/blog/5645.html
[17] https://www.semtech.cn/lora/lora-applications
[18] https://x-mol.com/offline
[19] https://cloud.tencent.com/developer/article/2103209
[20] https://www.ebyte.com/new-view-info.aspx?id=1066
[21] https://juejin.cn/post/7260897460613464122
[22] https://bdtechtalks.com/2023/05/22/what-is-lora/
[23] https://www.cdyxiot.com/xxzx_view-19-1326.html
[24] https://www.top-iot.com/content-26-490-1.html
[25] https://newsletter.x-mol.com/paper/1597640067431059456
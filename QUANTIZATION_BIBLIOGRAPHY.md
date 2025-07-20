# Comprehensive Bibliography: Neural Network Quantization

*A curated collection of key papers and resources in neural network quantization*

---

## Foundational Papers (2016-2018)

### Binary and Ternary Networks
1. **Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y.** (2016). "Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1." *arXiv preprint arXiv:1602.02830*.

2. **Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A.** (2016). "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks." *European Conference on Computer Vision*, 525-542.

3. **Li, F., Zhang, B., & Liu, B.** (2016). "Ternary Weight Networks." *arXiv preprint arXiv:1605.04711*.

4. **Zhu, C., Han, S., Mao, H., & Dally, W. J.** (2016). "Trained Ternary Quantization." *International Conference on Learning Representations*.

### Multi-bit Quantization
5. **Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., & Zou, Y.** (2016). "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients." *arXiv preprint arXiv:1606.06160*.

6. **Lin, D., Talathi, S., & Annapureddy, S.** (2016). "Fixed Point Quantization of Deep Convolutional Networks." *International Conference on Machine Learning*, 2849-2858.

7. **Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y.** (2017). "Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations." *Journal of Machine Learning Research*, 18(1), 6869-6898.

### Gradient Estimation
8. **Bengio, Y., Léonard, N., & Courville, A.** (2013). "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." *arXiv preprint arXiv:1308.3432*.

---

## Classical Methods (2018-2020)

### Post-Training Quantization
9. **Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., ... & Adam, H.** (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2704-2713.

10. **Krishnamoorthi, R.** (2018). "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper." *arXiv preprint arXiv:1806.08342*.

11. **Banner, R., Nahshan, Y., & Soudry, D.** (2019). "Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment." *Advances in Neural Information Processing Systems*, 7950-7958.

### Quantization-Aware Training
12. **Choi, J., Wang, Z., Venkataramani, S., Chuang, P. I. J., Srinivasan, V., & Gopalakrishnan, K.** (2018). "PACT: Parameterized Clipping Activation for Quantized Neural Networks." *arXiv preprint arXiv:1805.06085*.

13. **Jung, S., Son, C., Lee, S., Son, J., Han, J. J., Kwak, Y., ... & Choi, C.** (2019). "Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 4350-4359.

### Mixed-Precision Training
14. **Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Ginsburg, B.** (2017). "Mixed Precision Training." *International Conference on Learning Representations*.

15. **Wang, N., Choi, J., Brand, D., Chen, C., & Gopalakrishnan, K.** (2018). "Training Deep Neural Networks with 8-bit Floating Point Numbers." *Advances in Neural Information Processing Systems*, 7675-7684.

---

## Advanced Techniques (2020-2022)

### Adaptive Quantization
16. **Nagel, M., van Baalen, M., Blankevoort, T., & Welling, M.** (2020). "Up or Down? Adaptive Rounding for Post-Training Quantization." *International Conference on Machine Learning*, 7197-7206.

17. **Li, Y., Gong, R., Tan, X., Yang, Y., Hu, P., Zhang, Q., ... & Wang, H.** (2021). "BRECQ: Pushing the Limit of Post-Training Quantization by Block-wise Reconstruction." *International Conference on Learning Representations*.

18. **Wei, X., Gong, R., Li, Y., Liu, X., & Yu, F.** (2022). "QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization." *International Conference on Learning Representations*.

### Knowledge Distillation for Quantization
19. **Polino, A., Pascanu, R., & Alistarh, D.** (2018). "Model Compression via Distillation and Quantization." *International Conference on Learning Representations*.

20. **Mishra, A., & Marr, D.** (2017). "Apprentice: Using Knowledge Distillation Techniques to Improve Low-Precision Network Accuracy." *arXiv preprint arXiv:1711.05852*.

### Neural Architecture Search for Quantization
21. **Wang, K., Liu, Z., Lin, Y., Lin, J., & Han, S.** (2019). "HAQ: Hardware-Aware Automated Quantization with Mixed Precision." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 8612-8620.

22. **Wu, B., Wang, Y., Zhang, P., Tian, Y., Vajda, P., & Keutzer, K.** (2018). "Mixed Precision Quantization of ConvNets via Differentiable Neural Architecture Search." *arXiv preprint arXiv:1812.00090*.

---

## Large Language Model Era (2022-2024)

### Transformer Quantization
23. **Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L.** (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *Advances in Neural Information Processing Systems*, 35, 24438-24453.

24. **Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S.** (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *International Conference on Machine Learning*, 38087-38099.

25. **Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S.** (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *arXiv preprint arXiv:2306.00978*.

26. **Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D.** (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *arXiv preprint arXiv:2210.17323*.

### FP8 and Low-Precision Training
27. **Peng, H., Wu, K., Wei, Y., Zhao, G., Yang, Y., Liu, Z., ... & Cheng, P.** (2023). "FP8-LM: Training FP8 Large Language Models." *arXiv preprint arXiv:2310.18313*.

28. **Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P., Grisenthwaite, R., ... & Wu, H.** (2022). "FP8 Formats for Deep Learning." *arXiv preprint arXiv:2209.05433*.

### Extreme Quantization
29. **Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., ... & Wei, F.** (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." *arXiv preprint arXiv:2402.17764*.

30. **Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., ... & Wei, F.** (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." *arXiv preprint arXiv:2310.11453*.

---

## Survey Papers and Comprehensive Reviews

31. **Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., & Keutzer, K.** (2021). "A Survey of Quantization Methods for Efficient Neural Network Inference." *arXiv preprint arXiv:2103.13630*.

32. **Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., & Blankevoort, T.** (2021). "A White Paper on Neural Network Quantization." *arXiv preprint arXiv:2106.08295*.

33. **Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P.** (2020). "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation." *arXiv preprint arXiv:2004.09602*.

34. **Liu, Z., Shen, Z., Savvides, M., & Cheng, K. T.** (2020). "ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions." *European Conference on Computer Vision*, 143-159.

---

## Hardware-Aware Quantization

### GPU and Accelerator Optimization
35. **Jouppi, N. P., Young, C., Patil, N., Patterson, D., Agrawal, G., Bajwa, R., ... & Yoon, D. H.** (2017). "In-datacenter Performance Analysis of a Tensor Processing Unit." *ACM SIGARCH Computer Architecture News*, 45(2), 1-12.

36. **Chen, Y. H., Krishna, T., Emer, J. S., & Sze, V.** (2017). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks." *IEEE Journal of Solid-State Circuits*, 52(1), 127-138.

### Mobile and Edge Deployment
37. **Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H.** (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv preprint arXiv:1704.04861*.

38. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C.** (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520.

39. **Ma, N., Zhang, X., Zheng, H. T., & Sun, J.** (2018). "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design." *Proceedings of the European Conference on Computer Vision*, 116-131.

---

## Theoretical Foundations

### Information Theory and Rate-Distortion
40. **Agustsson, E., Mentzer, F., Tschannen, M., Cavigelli, L., Timofte, R., Benini, L., & Van Gool, L.** (2017). "Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations." *Advances in Neural Information Processing Systems*, 1141-1151.

41. **Theis, L., Shi, W., Cunningham, A., & Huszár, F.** (2017). "Lossy Image Compression with Compressive Autoencoders." *International Conference on Learning Representations*.

### Generalization Theory
42. **Xu, C., Yao, J., Lin, Z., Ou, W., Cao, Y., Wang, Z., & Zha, H.** (2018). "Alternating Multi-bit Quantization for Recurrent Neural Networks." *International Conference on Learning Representations*.

43. **Louizos, C., Reisser, M., Blankevoort, T., Gavves, E., & Welling, M.** (2019). "Relaxed Quantization for Discretized Neural Networks." *International Conference on Learning Representations*.

---

## Domain-Specific Applications

### Computer Vision
44. **Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., & Zhang, C.** (2017). "Learning Efficient Convolutional Networks through Network Slimming." *Proceedings of the IEEE International Conference on Computer Vision*, 2736-2744.

45. **Zhao, R., Hu, Y., Dotzel, J., De Sa, C., & Zhang, Z.** (2019). "Improving Neural Network Quantization without Retraining using Outlier Channel Splitting." *International Conference on Machine Learning*, 7543-7552.

### Natural Language Processing
46. **Shen, S., Dong, Z., Ye, J., Ma, L., Yao, Z., Gholami, A., ... & Keutzer, K.** (2020). "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT." *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(05), 8815-8821.

47. **Zhang, W., Hou, L., Yin, Y., Shang, L., Chen, X., Jiang, X., & Liu, Q.** (2020). "TernaryBERT: Distillation-aware Ultra-low Bit BERT." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 509-521.

### Speech and Audio
48. **Meller, E., Finkelstein, A., Almog, S., & Golan, M.** (2019). "Same, Same But Different: Recovering Neural Network Quantization Error Through Weight Factorization." *International Conference on Machine Learning*, 4486-4495.

---

## Emerging Techniques and Future Directions

### Differentiable Quantization
49. **Louizos, C., Ullrich, K., & Welling, M.** (2018). "Bayesian Compression for Deep Learning." *Advances in Neural Information Processing Systems*, 3288-3298.

50. **Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., & Modha, D. S.** (2019). "Learned Step Size Quantization." *International Conference on Learning Representations*.

### Dynamic and Adaptive Quantization
51. **Gong, R., Liu, X., Jiang, S., Li, T., Hu, P., Lin, J., ... & Wang, J.** (2019). "Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks." *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4852-4861.

52. **Yang, J., Shen, X., Xing, J., Tian, X., Li, H., Deng, B., ... & Huang, T. S.** (2019). "Quantization Networks." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 7308-7316.

### Quantum Computing and Quantization
53. **Schuld, M., Sinayskiy, I., & Petruccione, F.** (2015). "An Introduction to Quantum Machine Learning." *Contemporary Physics*, 56(2), 172-185.

54. **Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S.** (2017). "Quantum Machine Learning." *Nature*, 549(7671), 195-202.

---

## Software Frameworks and Tools

### PyTorch Quantization
55. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S.** (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems*, 8024-8035.

56. **Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P.** (2020). "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation." *arXiv preprint arXiv:2004.09602*.

### TensorFlow and TensorFlow Lite
57. **Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., ... & Zheng, X.** (2016). "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems." *arXiv preprint arXiv:1603.04467*.

58. **Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., ... & Adam, H.** (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2704-2713.

### ONNX and Cross-Platform Tools
59. **Bai, J., Lu, F., Zhang, K., et al.** (2019). "ONNX: Open Neural Network Exchange." *GitHub Repository*. Available: https://github.com/onnx/onnx

60. **Microsoft.** (2021). "ONNX Runtime: Cross-platform, High Performance ML Inferencing and Training Accelerator." Available: https://onnxruntime.ai/

---

## Benchmarks and Evaluation

### MLPerf and Standardized Benchmarks
61. **Mattson, P., Cheng, C., Coleman, C., Diamos, G., Micikevicius, P., Patterson, D., ... & Reddi, V. J.** (2020). "MLPerf Training Benchmark." *Proceedings of Machine Learning and Systems*, 2, 336-349.

62. **Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., Wu, C., ... & Young, M.** (2020). "MLPerf Inference Benchmark." *2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture*, 446-459.

### Language Model Benchmarks
63. **Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R.** (2018). "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding." *Proceedings of the 2018 EMNLP Workshop BlackboxNLP*, 353-355.

64. **Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., ... & Bowman, S. R.** (2019). "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems." *Advances in Neural Information Processing Systems*, 32, 3261-3275.

---

## Industry Applications and Case Studies

### Production Deployments
65. **Jouppi, N. P., Yoon, D. H., Ashcraft, M., Gottscho, M., Jablin, T. B., Kurian, G., ... & Patterson, D.** (2021). "Ten Lessons From Three Generations Deployed in Machine Learning Accelerators." *2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture*, 1-14.

66. **Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., ... & Guestrin, C.** (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." *13th USENIX Symposium on Operating Systems Design and Implementation*, 578-594.

### Mobile and Edge Case Studies
67. **Ignatov, A., Timofte, R., Kulik, A., Yang, S., Wang, K., Baum, F., ... & Van Gool, L.** (2019). "AI Benchmark: All About Deep Learning on Smartphones in 2019." *2019 IEEE/CVF International Conference on Computer Vision Workshop*, 3617-3635.

68. **Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S.** (2020). "Once-for-All: Train One Network and Specialize it for Efficient Deployment." *International Conference on Learning Representations*.

---

## Recent Advances (2023-2024)

### Latest Quantization Techniques
69. **Yao, Z., Aminabadi, R. Y., Zhang, M., Wu, X., Li, C., & He, Y.** (2022). "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers." *Advances in Neural Information Processing Systems*, 35, 27168-27183.

70. **Park, G., & Kwon, S. J.** (2023). "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs." *arXiv preprint arXiv:2402.04291*.

71. **Shao, W., Shi, M., Wang, S., Li, H., Chen, Y., Yao, A., & Tang, S.** (2023). "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models." *arXiv preprint arXiv:2308.13137*.

### Multimodal and Vision-Language Models
72. **Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad, Y., ... & Chandra, V.** (2023). "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models." *arXiv preprint arXiv:2305.17888*.

73. **Yuan, Z., Niu, L., Liu, J., Liu, Z., Shi, Y., Yuan, C., ... & Yan, S.** (2023). "RPTQ: Reorder-based Post-training Quantization for Large Language Models." *arXiv preprint arXiv:2304.01089*.

---

## Conference Proceedings and Journals

### Top-Tier Venues
- **NeurIPS** (Advances in Neural Information Processing Systems)
- **ICML** (International Conference on Machine Learning)
- **ICLR** (International Conference on Learning Representations)
- **CVPR** (IEEE Conference on Computer Vision and Pattern Recognition)
- **ICCV** (IEEE International Conference on Computer Vision)
- **ECCV** (European Conference on Computer Vision)
- **AAAI** (Association for the Advancement of Artificial Intelligence)
- **IJCAI** (International Joint Conference on Artificial Intelligence)

### Specialized Venues
- **MLSys** (Conference on Machine Learning and Systems)
- **ISCA** (International Symposium on Computer Architecture)
- **MICRO** (IEEE/ACM International Symposium on Microarchitecture)
- **DAC** (Design Automation Conference)
- **ASPLOS** (Architectural Support for Programming Languages and Operating Systems)

---

## Online Resources and Repositories

### Open Source Projects
74. **Hugging Face Transformers**: https://github.com/huggingface/transformers
75. **PyTorch Quantization**: https://pytorch.org/docs/stable/quantization.html
76. **TensorFlow Model Optimization**: https://www.tensorflow.org/model_optimization
77. **ONNX Runtime**: https://github.com/microsoft/onnxruntime
78. **Intel Neural Compressor**: https://github.com/intel/neural-compressor

### Datasets and Benchmarks
79. **ImageNet**: http://www.image-net.org/
80. **COCO**: https://cocodataset.org/
81. **OpenWebText**: https://github.com/jcpeterson/openwebtext
82. **Common Crawl**: https://commoncrawl.org/

---

*This bibliography represents a comprehensive collection of key papers and resources in neural network quantization as of January 2025. The field continues to evolve rapidly, and readers are encouraged to consult the latest publications for the most recent developments.*

**Last Updated**: January 2025  
**Total References**: 82+ papers and resources

**Abstract** 

In this project we have attempted to create a unique artistic image by composing two images where the new image will have content of one image and style of the other image. This is one of the kinds of painting where the artists have mastered their skills with huge practice over time. We have used Deep Neural Network (VGG Network) in order to accomplish the task. In this work, we can see an algorithmic approach of how humans create and perceive artistic imagery. 

**1    Problem statement** 

In view of creating new art in painting, artists always look to create something new. There is a type of painting they have mastered where they create a unique image by composing two  images such that the content of one image is shown in the style of the other image. We have worked on creating such artistic images using an approach mentioned in the research paper (provided in references), which utilises the properties of Deep Neural Network in order to achieve this task. Here, we combine the content and style of  two  different  images  using  neural  representation.  This  is  a  good  way  to  get  an  algorithmic understanding of how we create and perceive the artistic image.  This has already been used for a wide variety of applications and has a potential to open new possibilities especially in the field of graphic design and textile industry. 

**2    Proposed solution** 

In the research paper, the author has focussed on a class of Deep Neural Network which is very powerful for image processing named Convolutional Neural Networks. In CNN, we have layers of computational units where each unit processes some visual information or feature from the input image and output of the layer will be feature maps containing all the extracted features. We have used VGG - 19 architecture in our implementation. 

![](Project%20Report.001.png)

<center>Source: https://medium.com/machine-learning-algorithms/image-style-transfer-740d08f8c1bd </center>

Convolutional Neural Networks are used to train object recognition models. In this process, they develop a  representation  of  the  image  that  makes  object  information  increasingly  explicit  along  with  the processing hierarchy. Higher layers in the network capture the high-level content in terms of objects. However, lower layers simply reproduce the exact pixel values of the original image. So, we choose a feature map from a higher layer of the network for content representation. Content is nothing but a macro-structure of the input image.  

Using feature space designed for capturing texture information, we get a style representation of the input image. Visual pattern, textures, colors are the style of the input image. Content and style representation in CNN are separable but cannot be completely disentangled. Further, both of them can be manipulated separately to produce a new artistic or meaningful image.  

Inspired by deep learning algorithms, we minimize loss function for content and style during synthesis of new images. The loss function is given by: 

![](Project%20Report.002.png)

Here, α and β are user defined weighting parameters for reconstruction of content and style respectively. ![](Project%20Report.003.png) is photograph, ![](Project%20Report.004.png) is the artwork. The ratio of α /β can be in order of 10-1 to 10-5 

The content and style loss functions are mean squared error between squared error between input image and the generated image.  As discussed earlier, content loss is taken from the upper layer. However, the style layer loss is calculated over multiple layers of the network. The correlation of style loss is obtained by multiplying the feature map and its transpose, resulting in the gram matrix.  

![](Project%20Report.005.png)

<center> Image source: https://github.com/Adi-iitd/AI-Art </center>

To identify the style of image we compare different layers with their correlations. So, we are using the feature map gram matrix of each layer to obtain an image style. taking difference of the gram matrices and then difference of original and generated image gives us final style cost. Below is basic architecture. 

![](Project%20Report.006.png)

**3    Implementation details** 

- All the configurations required for the project are provided in a ‘config.ini’ file. 
- Pre-trained  layers  of  VGG19  are  read by the Tensorflow library. Thus, it is only necessary to download it once. The code takes care of this, if it isn’t downloaded before. 
- The content and the style image are stored in the data folder. Make sure to provide the correct name of these images in the config file 
- A copy of the content image is taken as the initial image, which over the course of iterations during the training process gets updated to the desired image with the style transferred from the style image. 
- Adam optimizer is used to train the model 
- Run ‘main.py’ file to get the results. 
- After the training completes, a display window shows along with a Trackbar, to see the evolution of the generated image over the course of the training process. 
- In addition, plots are generated for the style loss, content loss and the total loss over number of iterations. 
- For  the  purpose  of demonstration we have shown the working of the program on the same content image but two different style images. This is to verify that we are indeed able to transfer style from any style image onto the original image.** 

**4   Results and Discussion **

![](Project%20Report.007.png)

<center> Taking a content image and a style1 image </center>

​											 ![](Project%20Report.008.png)

<center> Content and style images taken Image obtained every 5 iterations </center>



​								![](Project%20Report.009.png)![](Project%20Report.010.png)![](Project%20Report.012.png)



![](Project%20Report.013.png)

<center> Final Image Generated </center>



Taking a content image and a style2 image

​																	 ![](Project%20Report.014.png)

<center> Content and style images taken </center>

​										 ![](Project%20Report.015.png)

<center> Image obtained every 5 iterations </center>

​															![](Project%20Report.016.png)![](Project%20Report.017.png)

![](Project%20Report.018.png)

![](Project%20Report.019.png)

<center> Final Image Generated </center>



- As it can be seen from the content\_loss graph, the content loss increases with the iterations 
- This is expected, as the initial image taken is the content image itself. Thus, it has the maximum content similarity before training.  
- But, the total loss decreases over time. This is because the style loss reduces over the training process 
- And since the style loss has much higher weight than the content loss (of the magnitude of 1e-5) the style loss causes the total loss to reduce over the iterations. 
- The final image generated primarily contains the content from the original image, but the style of the style image has been generated onto it. 

**5   References** 

- <https://arxiv.org/pdf/1508.06576.pdf>  
- [https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-t ransfer-ef88e46697ee](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee)  
- <https://towardsdatascience.com/neural-style-transfer-a-high-level-approach-250d4414c56b>  
- [https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-network s-7362f6cf4b9b](https://towardsdatascience.com/neural-style-transfer-and-visualization-of-convolutional-networks-7362f6cf4b9b)  
- <https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f>  
- <https://github.com/Adi-iitd/AI-Art> 
- <https://medium.com/machine-learning-algorithms/image-style-transfer-740d08f8c1bd>  

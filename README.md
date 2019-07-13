# FaceDetect

The modification is based on opencv repo: *https://github.com/opencv/opencv/*. The goal is to provide real-time prediction of letters by integrating a trained network into the facedetect.py code.

P.S. Before running the camshift algorithm, the model should be saved.

I rewrite the camshift class into a function "camshiftrun" so that it can run after the facedetect function. The face dected area will pass to the camshiftrun function so everything will work automatically.

We get the blocking area from the selection parameter and turn the probability of the this area to be zero. Otherwise, the areas of the hair and shoulder are also block.

Since we need to save the image based on the back projection image, the saving images process will be on when using back projection image.  I use a parameter "flag" to show whether the pattern has already saved.If the value of flag is zero, so you may can save the image. Also the aother pre-requirement is that the current trackbox is similiar to the previous trackbox. It means the camshift algorithm does not find a huge progress. The area is regarded as credible.

During the saving images process, calculate the truly hand area and resize the target image into 16*16 and 244*244 size to save them.



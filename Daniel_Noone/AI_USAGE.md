# Below is a general look at how I utilized AI in the formulation of my part of the project

1) First, I had to use it to understand and get the files using the AWS keys that we needed to get the MRI scan data
    I had never worked with this type of pipeline before, so it was informative to follow along and see how to get it working (even though it took awhile). This is likely the area I used it for most as Abraham provided helpful code as a guide, but I couldn't get it to work for some reason on my machine. I learned how to better work with large data repos on AWS, as this is something I am completely new to.
   
3) Then a major way I utilized AI was to help me understand the total focus and structure of the project
    I had never learned about GNNs or hypergraphs, and didn't understand these topics at first, but I was able to "start a dialogue" with GPT and Claude to better understand what I was looking at. I learned the high level concepts and the low level implementations better than if I had simply read a textbook or something, as I was able to ask the questions I specifically needed clarification on. 

4) Along with number 2 above, I also utilized it to understand how to translate the complex math operations I learned about into python, as I have stated, I haven't learned about these concepts before.

5) I also used AI to help me get the class labels, as I was confused as to where they were/how to read them. GPT allowed me to figure out how to quickly extract these. Initially it had the wrong feature names of the CSV data, but I was able to correct that easily.

6) Learning how to work with the monai package was a major hurdle as well that I used AI for, along with the select_t1_scan function in the preproc.py file
    I utilized AI to verify and help with using the package correctly, and it was also used for the function which selected the best T1 MRI scan from metadata. I didn't know how to go about all of that and have a better idea of how to walk through directories. Abraham also provided some helpful code as a guide for the select_t1_scan function which gave me a good starting point, but I needed to tweak it a little due to bugs in my specific envt. 




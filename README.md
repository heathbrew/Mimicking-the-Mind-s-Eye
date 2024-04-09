# Mimicking-the-Mind-s-Eye
AI-Driven Methodologies for Rorschach-Inspired Image Interpretation
![DALL·E 2023-11-29 23 10 58 - A Rorschach inkblot design featuring elements of a red crab and a red and pink flower, set against a white background that subtly transitions to a whi](https://github.com/heathbrew/Mimicking-the-Mind-s-Eye/assets/55629425/fb159d12-f9f8-449e-b514-b435e78bda28)
![DALL·E 2023-11-29 23 11 06 - A painting featuring a red crab holding a red and pink flower in its claw, set against a white background  The crab's face is detailed and prominent, ](https://github.com/heathbrew/Mimicking-the-Mind-s-Eye/assets/55629425/bd8236f5-9f00-401c-b6b4-7d3de4f53a5c)


Welcome to the Image Description Generation repository! Here, we present a comprehensive approach to generating detailed image descriptions using advanced artificial intelligence (AI) models. This methodology draws inspiration from the principles of the Rorschach Test, merging image captioning models with Visual Question Answering (VQA) systems to extract and combine detailed descriptions.

## Abstract
The rapid advancement of artificial intelligence has revolutionized image interpretation, mirroring the complexities explored in the Rorschach Test. Our research offers a cutting-edge approach that seamlessly combines image captioning models with VQA systems, allowing for intricate and nuanced image descriptions. By leveraging this methodology, we delve into concealed personality traits, confront biases, and enrich our understanding of human behaviour in various settings, thus contributing significantly to personality psychology.

## Proposed Methodology
Our methodology integrates three distinct AI models—Salesforce/blip, jaimin, and noamrot/FuseCap—for analyzing image descriptions. Through a comparative evaluation, we generate personality-informed descriptions for each Rorschach Test card, focusing on key aspects such as Content, Location, Determinants, Populars, and Form Quality.

## Usage
1. **Installation and Setup**: Follow the instructions in [install_absmergedesc.ps1](install_absmergedesc.ps1) to set up the Python virtual environment and install necessary dependencies.

2. **Running the Code**:
   - Execute the code in [RorshaqueimageDescription.ipynb](RorshaqueimageDescription.ipynb) for Rorchaque-based image description generation.
   - Utilize [imgdescbackend.py](imgdescbackend.py) to call the image captioning models ([noamrot/FuseCap_Image_Captioning](https://huggingface.co/noamrot/FuseCap_Image_Captioning), [jaimin/image_caption](https://huggingface.co/jaimin/image_caption), [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)) and the VQA model ([dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)).

## Additional Resources
- Read the Medium article on how to install and run Llama2 locally on Windows: [How to Install and Run Llama2 Locally on Windows for Free](https://medium.com/@AyushmanPranav/how-to-install-and-run-llama2-locally-on-windows-for-free-05bd5032c6e3?sk=8b8b54ccafd3f828bde4ab97c1913a28)
- Explore the code used for image description generation and model utilization in the respective folders inside the `Models` directory.

## Contribution
Contributions are encouraged and appreciated! Feel free to submit pull requests or open issues for any improvements or suggestions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
We extend our gratitude to the authors of the referenced AI models and libraries utilized in this research. Their contributions have been invaluable in shaping the methodology and outcomes of this study.

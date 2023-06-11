# Big Patent Summarization

## Overview
Welcome to the Big Patent Summarization project! This project aims to develop a modle that can efficiently summarize patent documents. By fine-tuning Hugging Face tokenizers and models, this project harnesses the state-of-the-art capabilities of NLP and ml to automate and streamline the process, making it more efficient and accurate. Shoutout to the authors of the paper for making the dataset available.

## Model
The summarization model used in Big Patent Summarization is a fine-tuned version of T5-small, trained specifically for patent text summarization tasks. After experimenting with different epochs and evaluating the loss graph, it was determined that training the model for 15 epochs produced optimal results. The trained model can be accessed on the Hugging Face Model Hub at [Jammal7/t5-small-finetuned-Big-Patents](https://huggingface.co/Jammal7/t5-small-finetuned-Big-Patents).

## Dataset
[BIGPATENT](https://arxiv.org/abs/1906.03741), consist of 1.3 million records of U.S. patent documents along with human written abstractive summaries. Each US patent application is filed under a Cooperative Patent Classification (CPC) code. The dataset is available on [Huggingface](https://huggingface.co/datasets/big_patent). The dataset had 9 classification categories:
- A: Human Necessities
- B: Performing Operations; Transporting
- C: Chemistry; Metallurgy
- D: Textiles; Paper
- E: Fixed Constructions
- F: Mechanical Engineering; Lighting; Heating; Weapons; Blasting Engines or Pumps
- G: Physics
- H: Electricity
- Y: General tagging of new or cross-sectional technology

For this project, I will be using the `D` category which is the Textiles; Paper category. The dataset is available in 3 different sizes: `small`, `medium` and `large`. I will be using the `small` dataset which has 10,000 train data; 560 validate&test records.

## Folder
1. `Data`: The data used during the course of the project.
2. `Notebook`: The notebooks used during the course of the project, including the tutorial notebook and the final notebook.
3. `src`: Some functions was defined as script and saved in this folder which was then called while training the model.

## Contributing
Contributions to the Big Patent Summarization project are welcome! If you would like to contribute, please follow these guidelines:

1. Fork the project repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with descriptive commit messages.
4. Push your branch to your forked repository.
5. Submit a pull request to the main project repository, outlining the changes you have


I hope that the Big Patent Summarization project proves to be a valuable tool for your patent analysis and research needs. As I continue working on this project, I plan to further enhance the model's efficiency by fine-tuning the model and increasing the training dataset. With increased access to GPU resources, I aim to develop an even more powerful and accurate model. If you have any questions, encounter issues, or would like to share your suggestions, please feel free to reach out to us. We are here to assist you. Happy patent summarization!
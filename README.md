# switrs-project

This is a Python-based project designed to generate structured prompts for testing large language models (LLMs). It uses geographic coordinates (latitude, longitude) within the Inland Empire (IE) region and integrates data from the StreetStory dataset (https://streetstory.berkeley.edu/) and SWITRS (2013-2023). The project evaluates LLMs' ability to make accurate predictions based on pure-text zero-shot prompts.

## Features
- Generates structured prompts using StreetStory and SWITRS datasets.
- Tests LLMs' predictive capabilities with zero-shot text-based prompts.

## Requirements
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Usage
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Modify coordinates in `main.py` and the prompt in `data_construct.py` to make changes.

## License
This project is licensed under the MIT License.

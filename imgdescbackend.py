from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer , ViTImageProcessor
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Define your model directories
model_dir1 = 'models/Salesforce-blip'
model_dir2 = 'models/jaimin'
model_dir3 = 'models/noamrot'
model_dir4 = 'models/vilt-b32-finetuned-vqa'

# Initialize models and processors
# BLIP model
blip_processor = BlipProcessor.from_pretrained(model_dir1)
blip_model = BlipForConditionalGeneration.from_pretrained(model_dir1)

# Vision-Encoder-Decoder model
ved_model = VisionEncoderDecoderModel.from_pretrained(model_dir2)
ved_feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir2)
ved_tokenizer = AutoTokenizer.from_pretrained(model_dir2)
ved_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Noamrot model
noamrot_processor = BlipProcessor.from_pretrained(model_dir3)
noamrot_model = BlipForConditionalGeneration.from_pretrained(model_dir3).to("cuda" if torch.cuda.is_available() else "cpu")

# VILT model
vilt_processor = ViltProcessor.from_pretrained(model_dir4)
vilt_model = ViltForQuestionAnswering.from_pretrained(model_dir4)

# Define functions for model predictions
def get_description_blip(image_path):
    image = Image.open(image_path)
    inputs = blip_processor(images=image, return_tensors="pt")
    description_ids = blip_model.generate(**inputs)
    description = blip_processor.decode(description_ids[0], skip_special_tokens=True)
    return description

def get_description_ved(image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    pixel_values = ved_feature_extractor(images=images, return_tensors="pt").pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
    output_ids = ved_model.generate(pixel_values, max_length=16, num_beams=4)
    preds = ved_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]


def get_description_noamrot(image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    pixel_values = noamrot_processor(images=images, return_tensors="pt").pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
    output_ids = noamrot_model.generate(pixel_values, num_beams=3, max_length=50)
    predictions = [noamrot_processor.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    return predictions

def perform_question_answering(image_path, questions):
    image = Image.open(image_path)
    answers_dict = {}
    for title, question in questions:
        encoding = vilt_processor(image, question, return_tensors="pt")
        outputs = vilt_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer_label = vilt_model.config.id2label[idx]
        answers_dict[title] = answer_label
    return answers_dict


def flatten_dictionary(nested_dict):
    flat_dict = {}
    for outer_key, inner_dict in nested_dict.items():
        if isinstance(inner_dict, dict):
            for inner_key, value in inner_dict.items():
                # Combine the outer and inner keys
                flat_key = f"{outer_key}_{inner_key}"
                flat_dict[flat_key] = value
        else:
            # If the value is not a dictionary, just copy it over
            flat_dict[outer_key] = inner_dict
    return flat_dict


def replace_keys_with_questions(input_dict):
    # Define the mapping from keys to questions
    key_to_question = {
        'answers_Content': ("Content", "What do you see in the image?"),
        'answers_Location': ("Location", "Where in the image does your attention focus the most?"),
        'answers_Determinants': ("Determinants", "What features or elements in the image influenced your perception?"),
        'answers_Populars': ("Populars", "Are there any common or recognizable elements in the image?"),
        'answers_Form Quality': ("Form Quality", "How would you describe the overall style or characteristics of the image?")
    }

    # Iterate over the dictionary and replace keys
    modified_dict = {}
    for key, value in input_dict.items():
        if key in key_to_question:
            # Replace key with the question text
            new_key = key_to_question[key][1]
            modified_dict[new_key] = value
        else:
            # Keep the key as is for other items
            modified_dict[key] = value

    return modified_dict

# Main function to process an image
def process_image(image_path):
    try:
        # Get descriptions from different models
        description_blip = get_description_blip(image_path)
        description_ved = get_description_ved([image_path])[0]
        description_noamrot = get_description_noamrot([image_path])[0]

        # Define questions for VILT model
        questions = [
            ("Content", "What do you see in the image?"),
            ("Location", "Where in the image does your attention focus the most?"),
            ("Determinants", "What features or elements in the image influenced your perception?"),
            ("Populars", "Are there any common or recognizable elements in the image?"),
            ("Form Quality", "How would you describe the overall style or characteristics of the image?"),
        ]

        # Get answers from VILT model
        answers = perform_question_answering(image_path, questions)

        # Aggregate results
        results = {
            "descriptions": {
                "blip": description_blip,
                "ved": description_ved,
                "noamrot": description_noamrot
            },
            "answers": answers
        }

        # return results

        flat_dict = flatten_dictionary(results)
        maindict = replace_keys_with_questions(flat_dict)
        
        return maindict
    
    except Exception as e:
        return {"error": str(e)}

# Example usage
# image_path = "static/uploads/example.jpg"
# result = process_image(image_path)
# print(result)

from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import torch

app = Flask(__name__)

# Load pretrained model and tokenizer
encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "ahmedabdo/facebook-bart-base-finetuned"
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(decoder_checkpoint).to('cpu')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    # Get the uploaded image
    uploaded_image = request.files['image']
    img = Image.open(uploaded_image).convert("RGB")
    features = feature_extractor(img, return_tensors="pt").pixel_values.to("cpu")
    print(features)

    # Generate caption
    caption = tokenizer.decode(model.generate(features, max_length=1024)[0], skip_special_tokens=True)
    print(caption)
    return jsonify({'caption': caption})

if __name__ == '_main_':
    app.run(debug=True)

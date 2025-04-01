import torch
import json
from transformers import pipeline
from extract_module import extract_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the JSON knowledge graph from the file
with open('E:/viva/project/rcm_sys/back_end/data/processing_disease_symptom_data.json', 'r', encoding="utf-8") as json_file:
    disease_dict = json.load(json_file)

# Load model directly
pipe_vi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")
pipe_en_vi = pipeline("translation", model="VietAI/envit5-translation")

# Function to find matching diseases based on symptoms
def recommend_diseases(symptoms, disease_data, top_n=3):
    disease_scores = {}
    true_symptoms = []
    tran_true_symptoms = []
    for disease, data in disease_data.items():
        count = data["Count of Disease Occurrence"]
        disease_symptoms = data["Symptoms"]

        # Calculate the matching score based on input symptoms
        match_score = sum(1 for symptom in symptoms if any(symptom.lower() in ds.lower() for ds in disease_symptoms))

        if match_score > 0:
            disease_scores[disease] = (count, match_score)
            for symptom in symptoms:
                if symptom in data["Symptoms"]:
                    if symptom not in true_symptoms:
                        true_symptoms.append(symptom)

    for true_symptom in true_symptoms:
        tran_true_symptom = pipe_en_vi("en: " + true_symptom)[0]["translation_text"]
        tran_true_symptoms.append(tran_true_symptom.strip("vi: "))

    # Sort diseases based on matching score and count of occurrence
    ranked_diseases = sorted(disease_scores.items(), key=lambda x: (-x[1][1], -x[1][0]))

    # Get the top N diseases
    top_diseases = ranked_diseases[:top_n]

    return top_diseases, tran_true_symptoms

# Generate a conversational response prompt for AI
def generate_prompt(recommended_diseases, true_symptoms):
    if not recommended_diseases:
        return "Dựa trên các triệu chứng được cung cấp, tôi không thể tìm thấy bất kỳ bệnh nào phù hợp. Hãy tham khảo ý kiến của chuyên gia chăm sóc sức khỏe để có chẩn đoán chính xác hơn."

    diseases = []

    for disease, (count, score) in recommended_diseases:
        tran_disease = pipe_en_vi("en: " + disease)[0]["translation_text"]
        diseases.append(tran_disease.strip("vi: "))

    if len(true_symptoms) == 1:
        symptoms_text = true_symptoms[0]
    else:
        symptoms_text = ', '.join(true_symptoms)

    response = f"Bạn có một vài triệu chứng bao gồm {symptoms_text}. Bệnh được chẩn đoán là {diseases[0]}, {diseases[1]} hoặc {diseases[2]}."

    return response


def inference(user_text):
    tran_user_text = pipe_vi_en(user_text)[0]["translation_text"]

    extract_symptoms = extract_info(user_text=tran_user_text)
    token_info = extract_symptoms.get_token_info()
    input_symptoms = extract_symptoms.extract_symptoms(token_info)

    recommended_diseases, true_symptoms = recommend_diseases(input_symptoms, disease_dict)
    prompt = generate_prompt(recommended_diseases, true_symptoms)

    return prompt
